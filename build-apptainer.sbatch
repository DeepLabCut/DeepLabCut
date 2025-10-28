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
DEF_FILE="$ROOT_DIR/DeepLabCut/conda-environments/dlc_cuda124.def"

mkdir -p "$IMG_DIR"

cp -f "$YAML" "$IMG_DIR/environment.yaml"
cp -f "$DEF_FILE" "$IMG_DIR/dlc_cuda124.def"

pushd "$IMG_DIR" >/dev/null
apptainer build --fakeroot "$IMG" "dlc_cuda124.def"
popd >/dev/null

echo "Built image: $IMG"
