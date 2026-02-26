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
"""Model Zoo utilities for in-memory inference entrypoints."""

from __future__ import annotations

from pathlib import Path

import torch

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.modelzoo.weight_initialization import build_weight_init
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    load_super_animal_config,
    update_config,
)


def create_superanimal_inference_runners(
    superanimal_name: str,
    model_name: str,
    detector_name: str | None = None,
    max_individuals: int = 10,
    batch_size: int = 1,
    detector_batch_size: int = 1,
    device: str | None = "auto",
    customized_model_config: str | Path | dict | None = None,
    customized_pose_checkpoint: str | Path | None = None,
    customized_detector_checkpoint: str | Path | None = None,
):
    """Create SuperAnimal inference runners for in-memory batched inference.

    This helper is intended for Model Zoo inference pipelines that run directly on
    arrays. It prepares pose/detector runners and returns them with the resolved
    model config.
    """
    if model_name.startswith("fmpose3d"):
        raise NotImplementedError(
            "FMPose3D is not supported in this helper. Use the FMPose3D inference API."
        )

    if superanimal_name == "superanimal_humanbody":
        raise NotImplementedError(
            "superanimal_humanbody is currently not supported by this helper because "
            "it relies on modelzoo.build_weight_init, which does not support this dataset."
        )

    if detector_name is None:
        raise ValueError(
            "Please provide `detector_name` for SuperAnimal top-down inference setup."
        )

    if device in (None, "auto"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if customized_model_config is not None:
        if isinstance(customized_model_config, (str, Path)):
            model_cfg = read_config_as_dict(customized_model_config)
        else:
            model_cfg = customized_model_config.copy()
    else:
        model_cfg = load_super_animal_config(
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
        )

    model_cfg = update_config(model_cfg, max_individuals=max_individuals, device=device)
    weight_init = build_weight_init(
        cfg=model_cfg,
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        with_decoder=False,
        memory_replay=False,
        customized_pose_checkpoint=customized_pose_checkpoint,
        customized_detector_checkpoint=customized_detector_checkpoint,
    )

    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=weight_init.snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        detector_path=weight_init.detector_snapshot_path,
    )

    return pose_runner, detector_runner, model_cfg
