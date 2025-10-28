#!/bin/bash
#SBATCH -J prep_appt_124
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"                                # adjust if needed
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
YAML="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"

# Use micromamba image WITH CUDA 12.4 + cuDNN9 pre-baked (no apt). Pick the closest 12.4.x tag available.
BASE="docker://mambaorg/micromamba:jammy-cuda-12.4.1"    # if 12.4.1 tag missing, use jammy-cuda-12.4.*
IMG="$IMG_DIR/micromamba_cuda124.sif"
OVL="$IMG_DIR/dlc-dev-env_124.ext3"

mkdir -p "$IMG_DIR"
apptainer version || true

# 1) Pull micromamba+CUDA image
if [ ! -f "$IMG" ]; then
  echo "Pulling $BASE ..."
  apptainer pull "$IMG" "$BASE"
fi

# 2) Create writable overlay (20 GB; adjust)
if [ ! -f "$OVL" ]; then
  dd if=/dev/zero of="$OVL" bs=1M count=0 seek=20000
  mkfs.ext3 -F "$OVL"
fi

# 3) Use micromamba (already in image) to build the env into the overlay
apptainer exec --nv \
  --bind "$SCRATCH:$SCRATCH" \
  --overlay "$OVL" \
  "$IMG" bash -lc '
set -euo pipefail
# Create env at a fixed prefix inside overlay; NO apt, micromamba is present
micromamba create -y -p /opt/conda/envs/DLC-DEV -f "'"$YAML"'"
# Checks
/opt/conda/envs/DLC-DEV/bin/python -V
/opt/conda/envs/DLC-DEV/bin/python - <<PY
import torch
print("torch", getattr(torch,"__version__",None), "cuda?", torch.cuda.is_available())
print("cuda runtime version:", getattr(getattr(torch, "version", None), "cuda", None))
PY
'
echo "Prepared:"
echo "  IMG=$IMG"
echo "  OVL=$OVL"
