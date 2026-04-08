import logging

from deeplabcut.pose_estimation_pytorch.data.postprocessor import build_detector_postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import build_bottom_up_preprocessor
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models.detectors.external import EXTERNAL_DETECTORS
from deeplabcut.pose_estimation_pytorch.runners import build_inference_runner
from deeplabcut.pose_estimation_pytorch.task import Task

logger = logging.getLogger(__name__)


def get_external_detector_inference_runner(
    detector_cfg: dict,
    batch_size: int,
    device: str,
    max_individuals: int,
    color_mode: str,
    transform=None,
    inference_cfg=None,
    min_bbox_score: float | None = None,
):
    if transform is None:
        transform = build_transforms({"scale_to_unit_range": True})

    detector = EXTERNAL_DETECTORS.build(detector_cfg)
    # to_device ?
    try:
        for param in detector.parameters():
            param.requires_grad = False
    except (AttributeError, RuntimeError):
        logger.warning(
            "External detector does not have parameters that can be frozen. "
            "Please review whether this is expected behavior for your detector."
        )
    try:
        detector.eval()
    except AttributeError:
        logger.warning(
            "External detector does not have an eval() method. "
            "Please review whether this is expected behavior for your detector."
        )

    runner = build_inference_runner(
        task=Task.DETECT,
        model=detector,
        device=device,
        snapshot_path=None,  # always pre-trained
        batch_size=batch_size,
        # NOTE: the "bottom-up preprocessor" is a bit of a misnomer for this use case as
        # this is a top-down pipeline, but what the preprocessor
        # does here is to load the images, augment them, and convert them to tensors,
        # which is what we need here as the input to the external detector.
        preprocessor=build_bottom_up_preprocessor(
            color_mode=color_mode,
            transform=transform,
        ),
        postprocessor=build_detector_postprocessor(
            max_individuals=max_individuals,
            min_bbox_score=min_bbox_score,
        ),
        inference_cfg=inference_cfg,
    )
    return runner
