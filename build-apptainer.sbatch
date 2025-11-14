#!/bin/bash
#SBATCH -J build_dlc_cuda124
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
IMG="$IMG_DIR/dlc_cuda124.sif"
YAML_SRC="${YAML_SRC:-$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml}"

DLC_SRC="${DLC_SRC:-$ROOT_DIR/DeepLabCut}"

mkdir -p "$IMG_DIR"
[[ -s "$YAML_SRC" ]] || { echo "ERROR: YAML not found: $YAML_SRC" >&2; exit 2; }

DEF="$IMG_DIR/dlc_cuda124.runtime.def"
cat > "$DEF" <<EOF
Bootstrap: docker
From: mambaorg/micromamba:jammy-cuda-12.4.1

%labels
    maintainer "Alek"
    purpose    "DeepLabCut CUDA 12.4 + cuDNN9 with baked conda env"

%setup
    mkdir -p "\${APPTAINER_ROOTFS}/opt/specs"
    cp "$YAML_SRC" "\${APPTAINER_ROOTFS}/opt/specs/environment.yaml"

%environment
    export PATH=/opt/envs/DLC-DEV/bin:\$PATH
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg

%post -c /bin/bash
    set -euo pipefail
    SPEC=/opt/specs/environment.yaml
    ls -l /opt/specs || true
    [[ -s "\$SPEC" ]]

    # Ensure no stray rc is read, but DO NOT use --no-rc
    rm -f /root/.condarc /etc/conda/condarc 2>/dev/null || true
    export MAMBA_ROOT_PREFIX=/opt/micromamba

    # Create env (channels passed explicitly)
    micromamba create -y \
      -c conda-forge -c pytorch -c nvidia \
      -p /opt/envs/DLC-DEV -f "\$SPEC"

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

apptainer build --fakeroot "$IMG" "$DEF"
echo "Built image: $IMG"
