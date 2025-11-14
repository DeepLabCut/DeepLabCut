import deeplabcut
import deeplabcut.compat
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
import yaml
from sklearn.model_selection import GroupKFold, KFold
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
import shutil
import datetime
import multiprocessing as mp
import sys
import json
from ruamel.yaml import YAML

# Number of parallel workers for running fold+seed combinations
# Set to 1 for sequential execution (useful for debugging)
# Set to higher values (e.g., 4, 8) to run multiple experiments in parallel
# Note: Each worker will use GPU resources, so adjust based on available GPU memory
N_WORKERS = 1

N_EPOCHS = 200
MODEL = 'resnet_50'
OUTPUT_STRIDE = 16
KEY_METRIC = 'test.rmse' #'test.mAP'
TRAIN_BATCH_SIZE = 16
# CUSTOM_WEIGHTS = '/home/alek/projects/cdl-test1/resnet50_unet_encoder_tuned.pth'

# 1. DEFINE CONFIGURATION AND PARAMETERS
# ---------------------------------------
# Set the full path to the project's config.yaml file
# IMPORTANT: Use an absolute path to avoid issues.
config_path = '/workspace/workdir/config.yaml'

# Check if the config file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"The specified config file does not exist: {config_path}\n"
        "Please update the 'config_path' variable with the correct path to your config.yaml file."
    )

# Number of folds for cross-validation
N_FOLDS = 4
N_SEEDS = 4
SHUFFLE_OFFSET = 100

# Note: Shuffle numbers are automatically assigned to prevent collisions
# Formula: shuffle_num = seed_idx * N_FOLDS + fold_idx + 1
# Example (4 folds, 2 seeds): Seed 0 uses shuffles 1-4, Seed 1 uses shuffles 5-8

# 2. MERGE DATA AND PREPARE FOR SPLITTING
# ------------------------------------------
# The `mergeandsplit` function ensures all labeled data is in one file.
# print("Merging annotated datasets...")
# deeplabcut.mergeandsplit(config_path, uniform=True)

# Read the merged data to get the total number of labeled frames.

def run_single_fold(args):
    """Run a single fold+seed combination. This function is designed to be run in parallel."""
    (seed_idx, fold_idx, train_indices, test_indices, config_path_template,
     experiment_id, group_by_video, train_overrides, landmark_sets,
     n_folds, n_seeds, num_frames, timestamp) = args

    # Create a unique config file for this fold+seed combination
    config_dir = Path(config_path_template).parent
    config_name = Path(config_path_template).stem
    config_ext = Path(config_path_template).suffix
    config_path = str(config_dir / f"{config_name}_seed{seed_idx}_fold{fold_idx}{config_ext}")

    # Copy the template config to the new location
    shutil.copy(config_path_template, config_path)

    try:
        # Use a unique shuffle number that incorporates both seed and fold
        # This prevents collisions when multiple seeds run in parallel
        # Formula: shuffle_num = seed_idx * n_folds + fold_idx + 1
        # Example: seed=0,fold=0 → shuffle=1; seed=0,fold=1 → shuffle=2
        #          seed=1,fold=0 → shuffle=5; seed=1,fold=1 → shuffle=6
        shuffle_num = seed_idx * n_folds + fold_idx + SHUFFLE_OFFSET
        print(f"\n\n{'='*20} FOLD {fold_idx+1}/{n_folds} SEED {seed_idx+1}/{n_seeds} (Shuffle {shuffle_num}) {'='*20}")
        train_fraction = round(len(train_indices) / num_frames, 2)
        print(f"Train ratio: {train_fraction:.2f}")

        train_fraction_percent = int(train_fraction * 100)

        # Read and update config file with new training fraction
        cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
        with open(config_path, 'r') as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw['TrainingFraction'] = [train_fraction]
        with open(config_path, 'w') as f:
            yaml.dump(cfg_raw, f)

        print(f"  Shuffle {shuffle_num}: Training with {len(train_indices)} frames, testing with {len(test_indices)} frames.")

        # b. Create the training dataset for this specific split
        print(f"  Creating training dataset for shuffle {shuffle_num}...")
        deeplabcut.create_training_dataset(
            config_path,
            Shuffles=[shuffle_num],
            trainIndices=[list(train_indices)],
            testIndices=[list(test_indices)],
            userfeedback=False,
            net_type=MODEL,
            augmenter_type='albumentations',
            # weight_init=WeightInitialization(CUSTOM_WEIGHTS) if CUSTOM_WEIGHTS else None
        )

        # override output_striFalsede (model.backbone.output_stride) and key_metric (runner.key_metric)
        # sample path: data/cdl-projects/test1-haag-2025-05-21/dlc-models-pytorch/iteration-0/test1May21-trainset75shuffle1/train/pytorch_config.yaml
        project_path = cfg['project_path']
        trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
        model_config_path = Path(project_path) / 'dlc-models-pytorch' / f'iteration-{cfg["iteration"]}' / trainingset_identifier / 'train' / 'pytorch_config.yaml'
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        for key, value in train_overrides.items():
            key_parts = key.split('.')
            current = model_cfg
            for part in key_parts[:-1]:
                current = current[part]
            current[key_parts[-1]] = value
        with open(model_config_path, 'w') as f:
            yaml.dump(model_cfg, f)

        # c. Train the network for this fold
        print(f"  Training network for shuffle {shuffle_num}...")
        # Adjust training parameters (e.g., maxiters) as needed.
        deeplabcut.train_network(
            config_path,
            shuffle=shuffle_num,
            max_snapshots_to_keep=2,
            autotune=False,
            displayiters=100,
            saveiters=5000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # d. Evaluate the trained network on the held-out test set
        print(f"  Evaluating network for shuffle {shuffle_num}...")
        evaluation_results = {}
        for l_idx, (landmark_set_name, landmark_set) in enumerate(landmark_sets.items()):
            iteration = cfg['iteration']
            engine_name = deeplabcut.compat.get_project_engine(cfg).aliases[0]
            trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
            evaluation_folder = Path(project_path) / f"evaluation-results-{engine_name}" / f"iteration-{iteration}" / trainingset_identifier
            # recursively delete evaluation folder contents, but not the folder itself
            if evaluation_folder.exists():
                for child in evaluation_folder.glob('*'):
                    if child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child)


            deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts=landmark_set)

            # e. Parse evaluation results and store them
            print(f"  Parsing evaluation results for shuffle {shuffle_num}...")
            # Construct the path to the evaluation folder

            # Find the results CSV file
            csv_files = list(evaluation_folder.glob('*-results.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No evaluation CSV file found in {evaluation_folder}")

            # Read the CSV and clean column names
            eval_df = pd.read_csv(csv_files[0])
            eval_df.columns = eval_df.columns.str.strip().str.replace('%', '') # Clean '%Training...'

            prefix_columns = ['test rmse', 'test rmse_pcutoff', 'test mAP', 'test mAR']

            if not eval_df.empty:
                # Convert the first row to a dictionary to get all columns
                summary_dict = eval_df.iloc[0].to_dict()
                summary_dict['fold'] = fold_idx # Add our custom fold number
                summary_dict['seed'] = seed_idx
                summary_dict['experiment'] = experiment_id
                # summary_dict['params'] = params_str
                summary_dict['group_by_video'] = group_by_video
                summary_dict['timestamp'] = timestamp
                for key, value in train_overrides.items():
                    summary_dict[f'override__{key}'] = value
                for col in prefix_columns:
                    summary_dict[f'{landmark_set_name}__{col}'] = summary_dict.pop(col)
                if l_idx == 0:
                    evaluation_results = summary_dict
                else:
                    evaluation_results.update(summary_dict)
            else:
                raise ValueError("Evaluation CSV file is empty.")

        return evaluation_resFalseults

    finally:
        # Clean up the temporary config file
        if os.path.exists(config_path):
            os.remove(config_path)




def run_experiment(config_path, n_folds, n_seeds, experiment_id='experiment_1', group_by_video=False, train_overrides={}, landmark_sets={'all': 'all'}):
    """Run cross-validation experiment with parallel processing of folds and seeds."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = deeplabcut.auxiliaryfunctions.read_config(config_path)
    project_path = cfg['project_path']
    trainingsetfolder = deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg)
    Data = merge_annotateddatasets(
                cfg,
                Path(os.path.join(project_path, trainingsetfolder)),
            )
    groups = np.array(list(map(lambda x: x[1], Data.axes[0])))
    num_frames = len(Data)
    print(f"Total number of labeled frames: {num_frames}")

    # Prepare all fold+seed combinations
    all_tasks = []
    for i in range(n_seeds):
        print(f"\n\n{'='*20} Preparing SEED {i+1}/{n_seeds} {'='*20}")

        if group_by_video:
            # Note: GroupKFold doesn't support random_state directly, so we shuffle groups manually
            unique_groups = np.unique(groups)
            rng = np.random.RandomState(42 + i)
            shuffled_group_order = rng.permutation(unique_groups)
            # Create a mapping from old group to new group based on shuffled order
            group_mapping = {old_g: new_g for new_g, old_g in enumerate(shuffled_group_order)}
            shuffled_groups = np.array([group_mapping[g] for g in groups])
            cv = GroupKFold(n_splits=n_folds)
            folds = list(cv.split(np.arange(num_frames), groups=shuffled_groups))
        else:
            cv = KFold(n_splits=n_folds, random_state=42+i, shuffle=True)
            folds = list(cv.split(np.arange(num_frames)))

        for j, (train_indices, test_indices) in enumerate(folds):
            task_args = (
                i,  # seed_idx
                j,  # fold_idx
                train_indices,
                test_indices,
                config_path,  # config_path_template
                experiment_id,
                group_by_video,
                train_overrides,
                landmark_sets,
                n_folds,
                n_seeds,
                num_frames,
                timestamp
            )
            all_tasks.append(task_args)

    print(f"\n\n{'='*20} Running {len(all_tasks)} tasks with {N_WORKERS} workers {'='*20}")

    # Run tasks in parallel
    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            evaluation_results_list = pool.map(run_single_fold, all_tasks)
    else:
        # Sequential execution for debugging
        evaluation_results_list = [run_single_fold(task) for task in all_tasks]

    # 5. AGGREGATE AND REPORT FINAL RESULTS
    # ------------------------------------
    print(f"\n\n{'='*20} Cross-Validation Summary {'='*20}")

    results_df = pd.DataFrame(evaluation_results_list)
    return results_df


if __name__ == "__main__":
    import uuid
    # skeletal_loss_weight: 0.0
    # skeletal_radius_multiplier_start: 1.15
    # skeletal_radius_multiplier_end: 1.15
    # union_intersect_adjacent_skeletal_mask_alpha_start: 0.0
    # union_intersect_adjacent_skeletal_mask_alpha_end: 0.0
    # union_intersect_adjacent_skeletal_mask_start_epoch: 0
    # union_intersect_adjacent_skeletal_mask_end_epoch: 1
    # use_skeletal_reference: true
    # truncate_targets: true
    # model.heads.bodypart.predictor.locref_std: 7.2801
    # model.heads.bodypart.target_generator.locref_std: 7.2801
    # model.heads.bodypart.target_generator.pos_dist_thresh: 17\
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    landmark_sets={
        'all':'all',
        'truncated': ['left_elbow', 'left_wrist', 'right_elbow', 'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'],
        'non_truncated': ['snout', 'base_of_head', 'left_shoulder', 'right_shoulder', 'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5', 'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3', 'tail4', 'tail5'],
        }
    experiments = [
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 1.10,
        #     'skeletal_radius_multiplier_end': 1.10,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': False,
        #     'truncate_targets': False,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'control',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 1.10,
        #     'skeletal_radius_multiplier_end': 1.10,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': False,
        #     'truncate_targets': False,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'control2',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 1.00,
        #     'skeletal_radius_multiplier_end': 1.00,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': True,
        #     'truncate_targets': True,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'xray_1.00',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 1.05,
        #     'skeletal_radius_multiplier_end': 1.05,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': True,
        #     'truncate_targets': True,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'xray_1.05',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 1.10,
        #     'skeletal_radius_multiplier_end': 1.10,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': True,
        #     'truncate_targets': True,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'xray_1.10',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 200.0,
        #     'skeletal_radius_multiplier_end': 200.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': True,
        #     'truncate_targets': True,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'xray_200',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_radius_multiplier_start': 0.80,
        #     'skeletal_radius_multiplier_end': 0.80,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': False,
        #     'truncate_targets': True,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'gt_0.80',
        #   'group_by_video': False,
        # },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.0,
        #     'skeletal_loss_radius_multiplier': 1.0,
        #     'skeletal_radius_multiplier_start': 0.80,
        #     'skeletal_radius_multiplier_end': 0.80,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': False,
        #     'truncate_targets': False,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'control',
        #   'group_by_video': False,
        # },
        {
            'train_overrides': {
            'skeletal_loss_weight': 0.1,
            'skeletal_loss_radius_multiplier': 1.0,
            'skeletal_radius_multiplier_start': 0.80,
            'skeletal_radius_multiplier_end': 0.80,
            'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
            'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
            'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
            'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
            'use_skeletal_reference': False,
            'truncate_targets': False,
            'model.heads.bodypart.predictor.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
            'runner.key_metric': 'test.rmse',
            'runner.key_metric_asc': False,
          },
          'experiment_id': 'll2_0.1_1.0',
          'group_by_video': False,
        },
        {
            'train_overrides': {
            'skeletal_loss_weight': 0.0,
            'skeletal_loss_radius_multiplier': 1.0,
            'skeletal_radius_multiplier_start': 1.15,
            'skeletal_radius_multiplier_end': 1.15,
            'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
            'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
            'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
            'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
            'use_skeletal_reference': True,
            'truncate_targets': True,
            'model.heads.bodypart.predictor.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.locref_std': 7.2801,
            'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
            'runner.key_metric': 'test.rmse',
            'runner.key_metric_asc': False,
          },
          'experiment_id': 'xray2_1.15',
          'group_by_video': False,
        },
        # {
        #     'train_overrides': {
        #     'skeletal_loss_weight': 0.1,
        #     'skeletal_loss_radius_multiplier': 1.1,
        #     'skeletal_radius_multiplier_start': 0.80,
        #     'skeletal_radius_multiplier_end': 0.80,
        #     'union_intersect_adjacent_skeletal_mask_alpha_start': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_alpha_end': 0.0,
        #     'union_intersect_adjacent_skeletal_mask_start_epoch': 0,
        #     'union_intersect_adjacent_skeletal_mask_end_epoch': 1,
        #     'use_skeletal_reference': False,
        #     'truncate_targets': False,
        #     'model.heads.bodypart.predictor.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.locref_std': 7.2801,
        #     'model.heads.bodypart.target_generator.pos_dist_thresh': 17,
        #     'runner.key_metric': 'test.rmse',
        #     'runner.key_metric_asc': False,
        #   },
        #   'experiment_id': 'll_0.1_1.1',
        #   'group_by_video': False,
        # },

    ]

    all_results = []
    config_filename = sys.argv[1]

    exp_cfg = None
    with open(config_filename, "r") as f:        
        exp_cfg = YAML().load(f)
    
    print (f"{config_path}")
    with open(config_path, "r") as f:        
        print(f.read())



    """
    for experiment in experiments:
        result_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results.to_csv(f'results_{result_timestamp}_{uuid.uuid4()}.csv')
        all_results.append(results)
    """
    results: pd.DataFrame = run_experiment(
            config_path, 
            N_FOLDS, 
            N_SEEDS, 
            exp_cfg['experiment_id'],
            group_by_video=exp_cfg['group_by_video'], 
            train_overrides=exp_cfg['train_overrides'], 
            landmark_sets=landmark_sets
        )

        
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(f'all_results_{timestamp}.csv')
