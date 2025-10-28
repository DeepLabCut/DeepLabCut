#!/bin/bash
#SBATCH -J build_dlc_cuda124
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -o %x_%j.out
set -euo pipefail

SCRATCH="${HOME}/scratch"
ROOT_DIR="$SCRATCH/dlc-experiments"
IMG_DIR="$ROOT_DIR/images"
DEF_SRC="$ROOT_DIR/DeepLabCut/conda-environments/dlc_cuda124.def"
YAML_SRC="$ROOT_DIR/DeepLabCut/conda-environments/DEEPLABCUT.yaml"
IMG="$IMG_DIR/dlc_cuda124.sif"

mkdir -p "$IMG_DIR"

# Copy inputs next to the build
cp -f "$DEF_SRC" "$IMG_DIR/dlc_cuda124.def"
cp -f "$YAML_SRC" "$IMG_DIR/environment.yaml"

# Build-time absolute path substitution for %files
ABS_YAML="$IMG_DIR/environment.yaml"
DEF_BUILD="$IMG_DIR/dlc_cuda124.build.def"
sed "s|environment.yaml /tmp/environment.yaml|$ABS_YAML /tmp/environment.yaml|" \
    "$IMG_DIR/dlc_cuda124.def" > "$DEF_BUILD"

pushd "$IMG_DIR" >/dev/null
apptainer build --fakeroot "$IMG" "$DEF_BUILD"
popd >/dev/null

echo "Built image: $IMG"
