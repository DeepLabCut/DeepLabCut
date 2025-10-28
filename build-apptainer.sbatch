#!/bin/bash
#SBATCH -J build_dlc_cuda124
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

# --- Paths (adjust if needed) ---
SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
IMG="$IMG_DIR/dlc_cuda124.sif"

# Your YAML (override by: sbatch --export=ALL,YAML_SRC=/path/to/env.yaml ...)
YAML_SRC="${YAML_SRC:-$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml}"

mkdir -p "$IMG_DIR"

# Validate YAML
if [[ ! -s "$YAML_SRC" ]]; then
  echo "ERROR: YAML not found or empty at: $YAML_SRC" >&2
  exit 2
fi

# Write a complete .def that uses %setup to copy the YAML into the image rootfs
DEF="$IMG_DIR/dlc_cuda124.runtime.def"
cat > "$DEF" <<EOF
Bootstrap: docker
From: mambaorg/micromamba:jammy-cuda-12.4.1
# If the exact tag is unavailable on your node, swap to :jammy-cuda-12.4 or :jammy

%labels
    maintainer "Alek"
    purpose    "DeepLabCut CUDA 12.4 + cuDNN9 with baked conda env"

# Copy the host YAML into the image's /tmp during build (runs on host)
%setup
    if [ ! -s "$YAML_SRC" ]; then
        echo "YAML not found: $YAML_SRC" >&2
        exit 2
    fi
    cp "$YAML_SRC" "\${APPTAINER_ROOTFS}/tmp/environment.yaml"

%environment
    export PATH=/opt/envs/DLC-DEV/bin:\$PATH
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg

# Build the env inside the image (runs inside container)
%post -c /bin/bash
    set -euo pipefail

    ls -l /tmp || true
    if [[ ! -s /tmp/environment.yaml ]]; then
        echo "ERROR: /tmp/environment.yaml missing or empty" >&2
        exit 2
    fi

    export MAMBA_ROOT_PREFIX=/opt/micromamba
    micromamba config set channels conda-forge,pytorch,nvidia
    micromamba config set channel_priority flexible

    micromamba create --no-rc -y -p /opt/envs/DLC-DEV -f /tmp/environment.yaml
    micromamba clean -a -y

    /opt/envs/DLC-DEV/bin/python -V
    /opt/envs/DLC-DEV/bin/python - <<'PY'
import torch
print("torch", getattr(torch,"__version__",None),
      "cuda?", bool(getattr(torch,"cuda",None) and torch.cuda.is_available()))
print("cuda runtime:", getattr(getattr(torch,"version",None),"cuda",None))
PY

%runscript
    exec "\$@"
EOF

# Build (fakeroot required to bake the env)
apptainer build --fakeroot "$IMG" "$DEF"

echo "Built image: $IMG"
