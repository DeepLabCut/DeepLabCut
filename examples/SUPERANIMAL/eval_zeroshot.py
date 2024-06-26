#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""SuperAnimal model zero-shot evaluation"""
from __future__ import annotations

from pathlib import Path

import torch

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset import TrainingDatasetMetadata
from deeplabcut.pose_estimation_pytorch import DLCLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluate import evaluate_snapshot
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.snapshots import Snapshot


def main(
    config_path: Path,
    super_animal: str,
    shuffle_index: int,
    device: str,
):
    metadata = TrainingDatasetMetadata.load(config_path, load_splits=True)
    shuffles = [s for s in metadata.shuffles if s.index == shuffle_index]
    if len(shuffles) != 1:
        raise ValueError(
            "Found multiple shuffles with different train indices but the same index "
            f"({shuffles}). To run this benchmark, there should only be one such "
            "shuffle."
        )

    shuffle = shuffles[0]
    print(f"Training shuffle: {shuffle.name}")
    print(f"  index: {shuffle.index}")
    print(f"  train fraction: {shuffle.train_fraction}")
    print(f"  train indices: {shuffle.split.train_indices}")
    print(f"  test indices: {shuffle.split.test_indices}")
    print()

    # edit config to have the desired training fraction
    af.edit_config(str(config_path), {"TrainingFraction": [shuffle.train_fraction]})

    # Load the config and create a data loader
    cfg = af.read_config(str(config_path))
    loader = DLCLoader(
        config=Path(cfg["project_path"]) / "config.yaml",
        shuffle=shuffle.index,
        trainset_index=0,
        modelprefix="",
    )
    loader.evaluation_folder.mkdir(exist_ok=True, parents=True)
    loader.model_cfg["device"] = device

    # Build the pose model
    model = PoseModel.build(
        loader.model_cfg["model"],
        weight_init=WeightInitialization.build(
            cfg=cfg,
            super_animal=super_animal,
            with_decoder=True,
        )
    )

    # Save the zero-shot snapshot
    state_dict = {
        "model": model.state_dict(),
        "metadata": {
            "epoch": 0,
            "metrics": {},
            "losses": {},
        },
    }
    snapshot_path = loader.model_folder / "zero-shot.pt"
    torch.save(state_dict, snapshot_path)

    # Evaluate the snapshot
    evaluate_snapshot(
        loader=loader,
        cfg=cfg,
        scorer=f"{super_animal}-zero-shot",
        snapshot=Snapshot(best=False, epochs=0, path=snapshot_path),
        transform=None,
        plotting=True,
        show_errors=True,
        detector_snapshot=None,
    )


if __name__ == "__main__":
    DATA = Path("/home/niels/datasets/superanimal")
    CONFIG_PATH = DATA / "openfield-Pranav-2018-08-20" / "config.yaml"
    SUPER_ANIMAL = "superanimal_topviewmouse"
    main(
        config_path=CONFIG_PATH,
        super_animal=SUPER_ANIMAL,
        shuffle_index=1001,
        device="cuda",
    )
