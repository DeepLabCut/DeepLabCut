#!/bin/bash
#SBATCH -J build-apptainer
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="$HOME/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"

IMG_DIR="$ROOT_DIR/images"
YAML="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"
BASE="docker://nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
IMG="$IMG_DIR/cuda.sif"
OVL="$IMG_DIR/dlc-dev-env.ext3"

mkdir -p "$IMG_DIR"

apptainer version || true

# 1) Pull CUDA 12.4 + cuDNN base
if [ ! -f "$IMG" ]; then
  echo "Pulling $BASE ..."
  apptainer pull "$IMG" "$BASE"
fi

# 2) Writable overlay for conda env (20 GB; adjust)
if [ ! -f "$OVL" ]; then
  dd if=/dev/zero of="$OVL" bs=1M count=0 seek=20000
  mkfs.ext3 -F "$OVL"
fi

# 3) Populate overlay: micromamba + env from YAML
apptainer exec --nv \
  --bind "$SCRATCH:$SCRATCH" \
  --overlay "$OVL" \
  "$IMG" bash -lc '
set -euo pipefail
apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git tini && rm -rf /var/lib/apt/lists/*
# micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
 | tar -xj -C /usr/local/bin bin/micromamba --strip-components=1
# create env at fixed prefix
micromamba create -y -p /opt/conda/envs/myenv -f "'"$YAML"'"
# sanity checks
/opt/conda/envs/myenv/bin/python -V
/opt/conda/envs/myenv/bin/python - <<PY
import torch, sys
print("torch", getattr(torch,"__version__",None), "cuda?", torch.cuda.is_available())
print("cuda runtime version:", torch.version.cuda if hasattr(torch,"version") else None)
PY
'
echo "Prepared:"
echo "  IMG=$IMG"
echo "  OVL=$OVL"
