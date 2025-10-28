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

BASE="docker://mambaorg/micromamba:jammy-cuda-12.4.1"
IMG="$IMG_DIR/micromamba_cuda124.sif"

ENVROOT="$ROOT_DIR/env_124"
mkdir -p "$IMG_DIR" "$ENVROOT"

[ -f "$IMG" ] || apptainer pull "$IMG" "$BASE"

apptainer exec --nv \
  --bind "$ENVROOT:/env","$SCRATCH:$SCRATCH" \
  "$IMG" bash -lc '
set -euo pipefail

# Put *everything* under /env and ignore user ~/.condarc
export MAMBA_ROOT_PREFIX=/env/mamba
export CONDA_PKGS_DIRS=/env/pkgs
export MAMBA_NO_HARD_LINKS=1
export MAMBA_NO_LOCK=1
export TMPDIR=/env/tmp
export XDG_CACHE_HOME=/env/.cache
export CONDARC=/env/condarc.yaml   # <- points to our clean rc
mkdir -p /env/{mamba,pkgs,envs,tmp,.cache}

# Minimal clean condarc (no YAML weirdness from $HOME)
cat >/env/condarc.yaml <<RC
channels:
  - conda-forge
  - pytorch
  - nvidia
channel_priority: flexible
auto_activate_base: false
ssl_verify: true
RC

# micromamba: skip any other rc files
umask 0002
ulimit -n 4096 || true

# Create env (explicitly tell micromamba to use only our rc)
micromamba create --no-rc -y -p /env/envs/DLC-DEV -f "'"$YAML"'"

# Sanity
/env/envs/DLC-DEV/bin/python -V
/env/envs/DLC-DEV/bin/python - <<PY
import torch
print("torch", getattr(torch,"__version__",None), "cuda?", torch.cuda.is_available())
print("cuda runtime:", getattr(getattr(torch,"version",None),"cuda",None))
PY
'
echo "OK. Env root = $ENVROOT"
