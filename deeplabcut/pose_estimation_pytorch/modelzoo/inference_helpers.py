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

import copy
import logging
from pathlib import Path

import deeplabcut.modelzoo.weight_initialization as weight_initialization
from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_filtered_coco_detector_inference_runner,
    get_inference_runners,
    get_pose_inference_runner,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    COCO_PERSON_CATEGORY_ID,
    get_super_animal_snapshot_path,
    load_super_animal_config,
    update_config,
)
from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
from deeplabcut.pose_estimation_pytorch.task import Task


def _build_humanbody_inference_runners(
    model_cfg: dict,
    model_name: str,
    detector_name: str | None,
    max_individuals: int,
    batch_size: int,
    detector_batch_size: int,
    customized_pose_checkpoint: str | Path | None,
    customized_detector_checkpoint: str | Path | None,
) -> tuple[InferenceRunner, InferenceRunner, dict]:
    if customized_detector_checkpoint is not None:
        logging.warning(
            "customized_detector_checkpoint is ignored for superanimal_humanbody. "
            "A filtered torchvision detector runner is used instead."
        )

    torchvision_detector_name = detector_name if detector_name is not None else "fasterrcnn_mobilenet_v3_large_fpn"

    pose_snapshot_path = customized_pose_checkpoint
    if pose_snapshot_path is None:
        pose_snapshot_path = get_super_animal_snapshot_path(
            dataset="superanimal_humanbody",
            model_name=model_name,
            download=True,
        )

    detector_runner = get_filtered_coco_detector_inference_runner(
        model_name=torchvision_detector_name,
        category_id=COCO_PERSON_CATEGORY_ID,
        batch_size=detector_batch_size,
        max_individuals=max_individuals,
        model_config=model_cfg,
    )
    pose_runner = get_pose_inference_runner(
        model_cfg,
        snapshot_path=pose_snapshot_path,
        batch_size=batch_size,
        max_individuals=max_individuals,
    )
    return pose_runner, detector_runner, model_cfg


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
        detector_name: Detector architecture name. For top-down SuperAnimal models,
            use detector names such as ``"fasterrcnn_resnet50_fpn_v2"``. For
            ``superanimal_humanbody``, this is interpreted as a torchvision detector
            name (default: ``"fasterrcnn_mobilenet_v3_large_fpn"``). Can be ``None``
            for bottom-up models.
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
        raise NotImplementedError("FMPose3D is not supported in this helper. Use the FMPose3D inference API.")

    if device is None:
        device = "auto"

    if customized_model_config is not None:
        if isinstance(customized_model_config, (str, Path)):
            model_cfg = read_config_as_dict(customized_model_config)
        else:
            model_cfg = copy.deepcopy(customized_model_config)
    else:
        model_cfg = load_super_animal_config(
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
        )
    model_cfg = update_config(model_cfg, max_individuals=max_individuals, device=device)

    if superanimal_name == "superanimal_humanbody":
        return _build_humanbody_inference_runners(
            model_cfg=model_cfg,
            model_name=model_name,
            detector_name=detector_name,
            max_individuals=max_individuals,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            customized_pose_checkpoint=customized_pose_checkpoint,
            customized_detector_checkpoint=customized_detector_checkpoint,
        )

    # Top-down models typically need a detector for bbox generation. If no detector
    # is configured, the returned detector_runner will be None and callers should
    # provide bboxes in the pose input context.
    if Task(model_cfg["method"]) == Task.TOP_DOWN and detector_name is None and customized_detector_checkpoint is None:
        logging.warning(
            "Top-down model configured without a detector. "
            "Returning detector_runner=None; pass bboxes in pose input context."
        )

    weight_init = weight_initialization.build_weight_init(
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
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        detector_path=weight_init.detector_snapshot_path,
    )
    return pose_runner, detector_runner, model_cfg
