#!/bin/bash
#SBATCH -J prep_appt_124
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
YAML="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"

# Micromamba + Ubuntu 22.04; CUDA 12.4 variant if available, else plain jammy works fine with --nv
BASE="docker://mambaorg/micromamba:jammy-cuda-12.4.1"   # if tag 404s: jammy-cuda-12.4  OR  jammy
IMG="$IMG_DIR/micromamba_cuda124.sif"
OVL="$IMG_DIR/dlc-dev-env_124.ext3"

mkdir -p "$IMG_DIR"

apptainer version || true

# 1) Pull micromamba image
[ -f "$IMG" ] || apptainer pull "$IMG" "$BASE"

# 2) Make a writable overlay (20 GiB)
if [ ! -f "$OVL" ]; then
  dd if=/dev/zero of="$OVL" bs=1M count=0 seek=20000
  mkfs.ext3 -F "$OVL"
fi

# 3) Build the env into the overlay with all conda/mamba paths under /env (writable)
apptainer exec --nv \
  --bind "$SCRATCH:$SCRATCH" \
  --overlay "$OVL" \
  "$IMG" bash -lc '
set -euo pipefail

# All writes go here (inside overlay)
export MAMBA_ROOT_PREFIX=/env
export CONDA_PKGS_DIRS=/env/pkgs
export MAMBA_NO_HARD_LINKS=1
export MAMBA_NO_LOCK=1

mkdir -p /env/pkgs /env/envs /opt/conda  # /opt/conda not used, but some tools expect it to exist
# Clean any previous broken state
micromamba clean --all --yes || true

# Create the env at a fixed prefix; no apt, no root needed
micromamba create -y -p /env/envs/DLC-DEV -f "'"$YAML"'"

# Sanity checks
/env/envs/DLC-DEV/bin/python -V
/env/envs/DLC-DEV/bin/python - <<PY
import torch, sys
print("torch", getattr(torch,"__version__",None), "cuda?", torch.cuda.is_available())
print("cuda runtime version:", getattr(getattr(torch, "version", None), "cuda", None))
PY
'
echo "Prepared:"
echo "  IMG=$IMG"
echo "  OVL=$OVL"
