import deeplabcut
import deeplabcut.compat
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import torch
import yaml
from deeplabcut.core.weight_init import WeightInitialization
from sklearn.model_selection import GroupKFold, KFold
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
import json



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
config_path = '/home/alek/projects/cdl-test1/data/cdl-projects/test1-haag-2025-05-21/config.yaml'

# Check if the config file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"The specified config file does not exist: {config_path}\n"
        "Please update the 'config_path' variable with the correct path to your config.yaml file."
    )

# Number of folds for cross-validation
k_folds = 4
n_repeat = 1

# 2. MERGE DATA AND PREPARE FOR SPLITTING
# ------------------------------------------
# The `mergeandsplit` function ensures all labeled data is in one file.
# print("Merging annotated datasets...")
# deeplabcut.mergeandsplit(config_path, uniform=True)

# Read the merged data to get the total number of labeled frames.

def run_experiment(config_path, n_folds, n_seeds, experiment_id, group_by_video=False, train_overrides={}, landmark_sets={'all': 'all'}):
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

    for i in range(n_seeds):
        print(f"\n\n{'='*20} SEED {i+1}/{n_seeds} {'='*20}")

        if group_by_video:
            cv = GroupKFold(n_splits=n_folds, random_state=42+i, shuffle=True)
            folds = cv.split(np.arange(num_frames), groups=groups)
        else:
            cv = KFold(n_splits=n_folds, random_state=42+i, shuffle=True)
            folds = cv.split(np.arange(num_frames))

        evaluation_results_list = []

        for j,(train_indices, test_indices)  in enumerate(folds):
            shuffle_num = j + 1
            print(f"\n\n{'='*20} FOLD {shuffle_num}/{n_folds} {'='*20}")
            train_fraction = round(len(train_indices) / num_frames, 2)
            print(f"Train ratio: {train_fraction:.2f}")

            train_fraction_percent = int(train_fraction * 100)


            # update config file with new training fraction
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

            # override output_stride (model.backbone.output_stride) and key_metric (runner.key_metric)
            # sample path: data/cdl-projects/test1-haag-2025-05-21/dlc-models-pytorch/iteration-0/test1May21-trainset75shuffle1/train/pytorch_config.yaml

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
            for l_idx, (landmark_set_name, landmark_set) in enumerate(landmark_sets.items()):   
                deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_num], plotting=False, comparisonbodyparts=landmark_set)

                # e. Parse evaluation results and store them
                print(f"  Parsing evaluation results for shuffle {shuffle_num}...")
                try:
                    # Construct the path to the evaluation folder
                    iteration = cfg['iteration']
                    engine_name = deeplabcut.compat.get_project_engine(cfg).aliases[0]
                    trainingset_identifier = f"{cfg['Task']}{cfg['date']}-trainset{train_fraction_percent}shuffle{shuffle_num}"
                    evaluation_folder = Path(project_path) / f"evaluation-results-{engine_name}" / f"iteration-{iteration}" / trainingset_identifier

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
                        summary_dict['fold'] = j # Add our custom fold numbere
                        summary_dict['seed'] = i
                        summary_dict['experiment'] = experiment_id
                        for col in prefix_columns:
                            summary_dict[f'{landmark_set_name}__{col}'] = summary_dict.pop(col)
                        if l_idx == 0:
                            evaluation_results_list.append(summary_dict)
                        else:
                            evaluation_results_list[-1].update(summary_dict)
                        print(f"  Fold {shuffle_num} - Test RMSE: {summary_dict.get('test rmse', 'N/A'):.2f} px")
                    else:
                        raise ValueError("Evaluation CSV file is empty.")

                except Exception as e:
                    print(f"  Could not find or parse evaluation file for shuffle {shuffle_num}. Error: {e}")


    # 5. AGGREGATE AND REPORT FINAL RESULTS
    # ------------------------------------
    print(f"\n\n{'='*20} Cross-Validation Summary {'='*20}")

    results_df = pd.DataFrame(evaluation_results_list)
    return results_df

        


if __name__ == "__main__":
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
    train_overrides={
        'skeletal_loss_weight': 0.0,
        'skeletal_radius_multiplier_start': 1.10,
        'skeletal_radius_multiplier_end': 1.10,
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
    }
    results: pd.DataFrame = run_experiment(config_path, k_folds, n_repeat, json.dumps(train_overrides | {'group_by_video': False}), group_by_video=False, train_overrides=train_overrides,
    landmark_sets={
        'all':'all',
        'truncated': ['left_elbow', 'left_wrist', 'right_elbow', 'right_wrist', 'left_knee', 'left_ankle', 'right_knee', 'right_ankle'],
        'non_truncated': ['snout', 'base_of_head', 'left_shoulder', 'right_shoulder', 'spine1', 'spine6', 'spine2', 'spine3', 'spine4', 'spine5', 'left_hip', 'right_hip', 'tail1', 'tail6', 'tail2', 'tail3', 'tail4', 'tail5'],
        })

    print(results)
    results.to_csv('results.csv')
