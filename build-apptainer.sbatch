#!/bin/bash
#SBATCH -J prep_bind_124
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
YAML="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"

# CUDA 12.4 micromamba base (if this tag 404s, use jammy-cuda-12.4 or just jammy)
BASE="docker://mambaorg/micromamba:jammy-cuda-12.4.1"
IMG="$IMG_DIR/micromamba_cuda124.sif"

# Env lives on the host FS here:
ENVROOT="$ROOT_DIR/env_124"      # <— persistent location in $SCRATCH

mkdir -p "$IMG_DIR" "$ENVROOT"

# Pull image once
[ -f "$IMG" ] || apptainer pull "$IMG" "$BASE"

# Build env into /env (host:$ENVROOT bound to container:/env)
apptainer exec --nv \
  --bind "$ENVROOT:/env","$SCRATCH:$SCRATCH" \
  "$IMG" bash -lc '
set -euo pipefail
# All conda/mamba writes go under /env (writable bind mount)
export MAMBA_ROOT_PREFIX=/env/mamba
export CONDA_PKGS_DIRS=/env/pkgs
export MAMBA_NO_HARD_LINKS=1
export MAMBA_NO_LOCK=1
export TMPDIR=/env/tmp
export XDG_CACHE_HOME=/env/.cache
mkdir -p /env/{mamba,pkgs,envs,tmp,.cache}

# Optional: pin channels for stability
micromamba config set channels conda-forge,pytorch,nvidia
micromamba config set channel_priority flexible

# Create env (no root needed)
micromamba create -y -p /env/envs/DLC-DEV -f "'"$YAML"'"

# Sanity
/env/envs/DLC-DEV/bin/python -V
/env/envs/DLC-DEV/bin/python - <<PY
import torch
print("torch", getattr(torch,"__version__",None), "cuda?", torch.cuda.is_available())
print("cuda runtime:", getattr(getattr(torch,"version",None),"cuda",None))
PY
'
echo "OK. Env root = $ENVROOT"
