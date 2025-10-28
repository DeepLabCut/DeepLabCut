#!/bin/bash
#SBATCH -J build_dlc_cuda124
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
DEF="$IMG_DIR/dlc_cuda124.def"
IMG="$IMG_DIR/dlc_cuda124.sif"
YAML="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"

mkdir -p "$IMG_DIR"

# Write the Apptainer definition file
cat > "$DEF" <<'DEF'
Bootstrap: docker
From: mambaorg/micromamba:jammy-cuda-12.4.1
# If this tag is unavailable on your system, change to jammy-cuda-12.4 or jammy.

%files
    environment.yaml /tmp/environment.yaml

%environment
    export PATH=/opt/envs/DLC-DEV/bin:$PATH
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg

%post
    set -euo pipefail
    # Keep micromamba state local to the image during build
    export MAMBA_ROOT_PREFIX=/opt/micromamba
    micromamba config set channels conda-forge pytorch nvidia
    micromamba config set channel_priority flexible
    micromamba create -y -p /opt/envs/DLC-DEV -f /tmp/environment.yaml
    micromamba clean -a -y

    # Smoke test
    /opt/envs/DLC-DEV/bin/python -V
    /opt/envs/DLC-DEV/bin/python - <<PY
import torch, sys
print("torch", getattr(torch,"__version__",None), "cuda?", getattr(torch,"cuda",None) and torch.cuda.is_available())
print("cuda runtime:", getattr(getattr(torch,"version",None),"cuda",None))
PY

%runscript
    # Default: run whatever the user passes (python, bash, etc.)
    exec "$@"
DEF

# Copy your YAML into the same dir as the .def so %files can include it
cp -f "$YAML" "$IMG_DIR/environment.yaml"

# Build the image with fakeroot (no overlay, no host binds required for the env)
# If this fails with "fakeroot not permitted", you'll need admins to enable it.
apptainer build --fakeroot "$IMG" "$DEF"

echo "Built image: $IMG"
