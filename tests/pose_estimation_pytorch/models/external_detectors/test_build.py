import numpy as np

from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
    build_detector_postprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
    build_bottom_up_preprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models.detectors.external import (
    EXTERNAL_DETECTORS,
)
from deeplabcut.pose_estimation_pytorch.runners.inference import build_inference_runner
from deeplabcut.pose_estimation_pytorch.task import Task


def test_external_detector_end_to_end_inference():
    """
    End-to-end test for external (pretrained) detector inference.

    This test verifies that:
    - an external detector can be built from the registry
    - preprocessing runs correctly
    - DetectorInferenceRunner executes inference
    - outputs have the expected DLC detection structure
    """

    # -------------------------
    # 1. Build the external detector
    # -------------------------
    detector_cfg = {
        "type": "MockExternalDetector",
        "score": 0.9,
        "label": 1,
    }

    detector = EXTERNAL_DETECTORS.build(detector_cfg)
    detector.eval()

    # -------------------------
    # 2. Build preprocessor & postprocessor
    # -------------------------
    transform = build_transforms({"scale_to_unit_range": True})

    preprocessor = build_bottom_up_preprocessor(
        color_mode="RGB",
        transform=transform,
    )

    postprocessor = build_detector_postprocessor(
        max_individuals=5,
        min_bbox_score=0.0,
    )

    # -------------------------
    # 3. Build inference runner (high-level API)
    # -------------------------
    runner = build_inference_runner(
        task=Task.DETECT,
        model=detector,
        device="cpu",
        snapshot_path=None,  # external detectors manage their own weights
        batch_size=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    # -------------------------
    # 4. Create mock input data
    # -------------------------
    # Single RGB image (H, W, C)
    image = np.zeros((128, 256, 3), dtype=np.uint8)

    # -------------------------
    # 5. Run inference
    # -------------------------
    results = runner.inference([image])

    # -------------------------
    # 6. Check outputs
    # -------------------------
    assert isinstance(results, list)
    assert len(results) == 1

    det = results[0]
    assert isinstance(det, dict)
    assert "bboxes" in det
    assert "bbox_scores" in det

    bboxes = det["bboxes"]
    scores = det["bbox_scores"]

    assert isinstance(bboxes, np.ndarray)
    assert isinstance(scores, np.ndarray)

    assert bboxes.shape == (1, 4)
    assert scores.shape == (1,)

    # Check bbox sanity (MockExternalDetector returns centered box)
    x1, y1, x2, y2 = bboxes[0]
    assert x2 > x1
    assert y2 > y1

    # Score sanity
    assert np.isclose(scores[0], 0.9)
