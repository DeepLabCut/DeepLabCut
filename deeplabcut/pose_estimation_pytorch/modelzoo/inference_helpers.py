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
"""PyTorch-specific helper entrypoints for model zoo inference."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    load_super_animal_config,
    update_config,
)
from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
from deeplabcut.pose_estimation_pytorch.task import Task


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
) -> tuple[InferenceRunner, InferenceRunner | None, dict]:
    """Create SuperAnimal inference runners for in-memory batched inference.

    This helper is intended for Model Zoo inference pipelines that run directly on
    arrays. It prepares pose/detector runners and returns them with the resolved
    model config.

    Args:
        superanimal_name: Name of the SuperAnimal dataset, e.g.
            ``"superanimal_quadruped"``.
        model_name: Pose model architecture name, e.g. ``"hrnet_w32"``.
        detector_name: Detector architecture name for top-down inference, e.g.
            ``"fasterrcnn_resnet50_fpn_v2"``. Can be ``None`` for bottom-up models.
        max_individuals: Maximum number of individuals to keep per frame.
        batch_size: Batch size for pose inference.
        detector_batch_size: Batch size for detector inference.
        device: Device for inference. If ``"auto"`` or ``None``, resolves to CUDA
            when available, else CPU.
        customized_model_config: Optional path or dict for a custom model config.
            If not provided, uses the default SuperAnimal config. Note that this config
            determines whether the model is top-down or bottom-up; for bottom-up models,
            ``detector_runner`` will be ``None`` even if ``detector_name`` is set.
        customized_pose_checkpoint: Optional custom pose checkpoint path.
        customized_detector_checkpoint: Optional custom detector checkpoint path.

    Returns:
        tuple: ``(pose_runner, detector_runner, model_cfg)`` where:
            - ``pose_runner`` is the pose inference runner
            - ``detector_runner`` is the detector inference runner or ``None`` if no
              detector is configured
            - ``model_cfg`` is the resolved model configuration dict

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from PIL import Image
        >>> from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
        ...     create_superanimal_inference_runners,
        ... )
        >>>
        >>> img_paths = [
        ...     "/path/to/images/frame_0000.png",
        ...     "/path/to/images/frame_0001.png",
        ...     "/path/to/images/frame_0002.png",
        ... ]
        >>> images = [np.asarray(Image.open(Path(p)).convert("RGB")) for p in img_paths]
        >>>
        >>> pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
        ...     superanimal_name="superanimal_quadruped",
        ...     model_name="hrnet_w32",
        ...     detector_name="fasterrcnn_resnet50_fpn_v2",
        ...     max_individuals=10,
        ...     batch_size=1,
        ...     detector_batch_size=1,
        ... )
        >>>
        >>> det_preds = det_runner.inference(images) if det_runner is not None else None
        >>> pose_inputs = list(zip(images, det_preds)) if det_preds is not None else images
        >>> pose_preds = pose_runner.inference(pose_inputs)
        >>> print(len(pose_preds))
    """
    if model_name.lower().startswith("fmpose3d"):
        raise NotImplementedError(
            "FMPose3D is not supported in this helper. Use the FMPose3D inference API."
        )

    if superanimal_name == "superanimal_humanbody":
        raise NotImplementedError(
            "superanimal_humanbody is currently not supported by this helper because "
            "it relies on modelzoo.build_weight_init, which does not support this dataset."
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

    # Top-down models typically need a detector for bbox generation. If no detector
    # is configured, the returned detector_runner will be None and callers should
    # provide bboxes in the pose input context.
    if (
        Task(model_cfg["method"]) == Task.TOP_DOWN
        and detector_name is None
        and customized_detector_checkpoint is None
    ):
        logging.warning(
            "Top-down model configured without a detector. "
            "Returning detector_runner=None; pass bboxes in pose input context."
        )

    model_cfg = update_config(model_cfg, max_individuals=max_individuals, device=device)
    # Local import avoids import-time circular dependency with modelzoo package init.
    from deeplabcut.modelzoo.weight_initialization import build_weight_init

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
