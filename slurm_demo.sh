#!/bin/bash

#SBATCH --partition=gpu             # Partition (job queue)
#SBATCH --no-requeue                 # Do not re-run job  if preempted
#SBATCH --job-name=dlc_p100_gpu1            # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH --constraint=pascal		# specifies to use the Pascal Node which has the P100 GPU
#SBATCH --nodelist=pascal005
#SBATCH --mem=50000                  # Real memory (RAM) required (MB)
#SBATCH --time=30:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.p100_gpu1.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.p100_gpu1.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

module purge
source activate dlc
module use /projects/community/modulefiles
module load cuda/9.0 cudnn
export DLClight=True

python dlc_script.py
