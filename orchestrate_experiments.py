import yaml
import os
import shutil
import uuid
import copy

N_FOLDS = 8
N_SEEDS = 1
SEED_OFFSET = 0
CODE_BRANCH = 'main'

sbatch_template = './run_experiment.sbatch'
data_archive_template = '$HOME/scratch/dlc-experiments/data/partial_project.zip'
config_path_template = '$HOME/scratch/dlc-experiments/experiment_configs/{config_name}'
apptainer_path = '$HOME/scratch/dlc-experiments/images/dlc_cuda124.sif'

experiments_log_dir = '~/scratch/dlc-experiments/logs'
os.makedirs(experiments_log_dir, exist_ok=True)

experiment_configs_dir = '~/scratch/dlc-experiments/experiment_configs'
os.makedirs(experiment_configs_dir, exist_ok=True)

base_config = {
            'train_overrides': {
            'skeletal_loss_weight': 0.0,
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
          'experiment_id': 'control',
          'group_by_video': False,
          'n_folds': N_FOLDS,
          'n_seeds': N_SEEDS,
          'seed_idx': 0,
          'fold_idx': 0,
          'seed_offset': SEED_OFFSET,
          'job_guid': ''
        },

username = os.environ['USER']

for seed_idx in range(N_SEEDS):
    for fold_idx in range(N_FOLDS):
        job_name =f'{base_config["experiment_id"]}_seed{seed_idx}_fold{fold_idx}'
        config = copy.deepcopy(base_config)
        config['seed_idx'] = seed_idx
        config['fold_idx'] = fold_idx
        job_guid = uuid.uuid4()
        config['job_guid'] = job_guid

        config_filename = f'{job_guid}.yaml'
        sbatch_filename = f'{job_guid}.sbatch'

        with open(os.path.join(experiment_configs_dir, config_filename), 'w') as f:
            yaml.dump(config, f)

        # populate sbatch template
        with open(sbatch_template, 'r') as f:
            sbatch = f.read()
            sbatch = sbatch.replace('{{JOB_NAME_PLACEHOLDER}}', job_name)
            sbatch = sbatch.replace('{{GATECH_USERNAME_PLACEHOLDER}}', username)
            sbatch = sbatch.replace('{{DATA_ARCHIVE_PLACEHOLDER}}', data_archive_template)
            sbatch = sbatch.replace('{{CONFIG_FILE_PLACEHOLDER}}', config_path_template.format(config_name=config_filename))
            sbatch = sbatch.replace('{{APPTAINER_PATH_PLACEHOLDER}}', apptainer_path)
            sbatch = sbatch.replace('{{CODE_BRANCH_PLACEHOLDER}}', CODE_BRANCH)

        sbatch_filepath = os.path.join(experiment_configs_dir, sbatch_filename)

        with open(sbatch_filepath, 'w') as f:
            f.write(sbatch)

        # set CWD to log dir
        os.chdir(experiments_log_dir)

        # submit job
        os.system(f'sbatch {sbatch_filepath}')
