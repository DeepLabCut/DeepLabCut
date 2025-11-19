#!/bin/bash
#SBATCH --job-name={{JOB_NAME_PLACEHOLDER}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH -t04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={{GATECH_USERNAME_PLACEHOLDER}}@gatech.edu
#SBATCH -o %x_%j.out
#SBATCH --tmp=10G
#SBATCH --gres=gpu:1

DATA_ARCHIVE={{DATA_ARCHIVE_PLACEHOLDER}}
EXPERIMENT_CONFIG_FILE={{CONFIG_FILE_PLACEHOLDER}}
APPTAINER_PATH={{APPTAINER_PATH_PLACEHOLDER}}
CODE_BRANCH={{CODE_BRANCH_PLACEHOLDER}}

WORKDIR=/tmp/dlc_project
CODEDIR=/tmp/DeepLabCut
mkdir -p "$WORKDIR" "$CODEDIR"

mkdir -p $WORKDIR/dlc-models-pytorch
mkdir -p $WORKDIR/dlc-models
mkdir -p $WORKDIR/evaluation-results-pytorch
mkdir -p $WORKDIR/training-datasets
mkdir -p $WORKDIR/videos
mkdir -p $WORKDIR/processed

## Extract project files from archive and shift labelled data and config.yaml upwards
unzip -q $DATA_ARCHIVE -d $WORKDIR
mv $WORKDIR/data_archive/labeled-data $WORKDIR
mv $WORKDIR/data_archive/dlc-models-pytorch $WORKDIR
mv $WORKDIR/data_archive/training-datasets $WORKDIR
mv $WORKDIR/data_archive/config.yaml $WORKDIR
## Clone DLC
#git clone https://github.com/bit-wrangler/DeepLabCut.git $CODEDIR
#cd $CODEDIR
#git checkout $CODE_BRANCH
cp -r ~/scratch/dlc-experiments/DeepLabCut /tmp/

cd $CODEDIR
## Copy config file to working directory
cp $EXPERIMENT_CONFIG_FILE $WORKDIR/experiment_cfg.yaml
echo $WORKDIR 
ls $WORKDIR
## Run experiment via apptainer
apptainer exec --nv \
  --bind "$CODEDIR":/workspace/DeepLabCut \
  --bind "$WORKDIR":/workspace/workdir \
  --pwd /workspace/DeepLabCut \
  "$APPTAINER_PATH" \
  bash -lc '
    export PYTHONPATH=/workspace/DeepLabCut:$PYTHONPATH
    python run_experiments_with_cv.py /workspace/workdir/experiment_cfg.yaml
  '

