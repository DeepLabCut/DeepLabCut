from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.data.preprocessor import build_top_down_preprocessor
from deeplabcut.pose_estimation_pytorch.runners.inference import (
    DetectorToPoseInferenceRunner,
    build_inference_runner,
)
from deeplabcut.pose_estimation_pytorch.task import Task


class DummyDetectorRunner:
    """Simple detector runner stub returning predefined outputs."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def inference(self, images, shelf_writer=None):
        images = list(images)
        self.calls.append(
            {
                "images": images,
                "shelf_writer": shelf_writer,
            }
        )
        return self.outputs


class RecordingPoseRunner:
    """
    Minimal pose runner stub that records what it receives and returns a fixed result.
    """

    def __init__(self, return_value=None):
        self.calls = []
        self.return_value = return_value if return_value is not None else [{"ok": True}]

    def inference(self, images, shelf_writer=None):
        images = list(images)
        self.calls.append(
            {
                "images": images,
                "shelf_writer": shelf_writer,
            }
        )
        return self.return_value


class PreprocessingPoseRunner:
    """
    Small integration-style pose runner that actually runs the real top-down preprocessor.

    This lets us verify that the wrapper injects context["bboxes"] in the exact form
    expected by TopDownCrop.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.calls = []

    def inference(self, images, shelf_writer=None):
        images = list(images)
        self.calls.append(
            {
                "images": images,
                "shelf_writer": shelf_writer,
            }
        )

        outputs = []
        for item in images:
            if isinstance(item, tuple):
                image, context = item
            else:
                image, context = item, {}

            proc_image, proc_context = self.preprocessor(image, context)

            outputs.append(
                {
                    "image_shape": tuple(proc_image.shape),
                    "num_bboxes": len(context["bboxes"]),
                    "offsets_shape": tuple(np.asarray(proc_context["offsets"]).shape),
                    "scales_shape": tuple(np.asarray(proc_context["scales"]).shape),
                    "top_down_crop_size": proc_context["top_down_crop_size"],
                }
            )

        return outputs


def test_detector_then_pose_inference_injects_bboxes_and_preserves_context():
    detector_outputs = [
        {
            "bboxes": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
            "bbox_scores": np.array([0.9], dtype=np.float32),
        },
        {
            "bboxes": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                dtype=np.float32,
            ),
            "bbox_scores": np.array([0.7, 0.8], dtype=np.float32),
        },
    ]

    detector_runner = DummyDetectorRunner(detector_outputs)
    pose_runner = RecordingPoseRunner(return_value=[{"poses": "ok"}])

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    original_context0 = {"foo": "bar"}
    original_context1 = {"answer": 42}

    images = [
        ("img0.png", original_context0),
        (Path("img1.png"), original_context1),
    ]

    results = runner.inference(images)

    assert results == [{"poses": "ok"}]

    # Detector got the original inputs
    assert len(detector_runner.calls) == 1
    assert detector_runner.calls[0]["images"] == images

    # Pose runner got enriched inputs
    assert len(pose_runner.calls) == 1
    enriched = pose_runner.calls[0]["images"]
    assert len(enriched) == 2

    image0, context0 = enriched[0]
    assert image0 == "img0.png"
    assert context0["foo"] == "bar"
    np.testing.assert_allclose(
        context0["bboxes"],
        np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        context0["bbox_scores"],
        np.array([0.9], dtype=np.float32),
    )
    assert context0["detector_output"] is detector_outputs[0]

    image1, context1 = enriched[1]
    assert image1 == Path("img1.png")
    assert context1["answer"] == 42
    np.testing.assert_allclose(
        context1["bboxes"],
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        context1["bbox_scores"],
        np.array([0.7, 0.8], dtype=np.float32),
    )
    assert context1["detector_output"] is detector_outputs[1]

    # Original input contexts should remain untouched
    assert original_context0 == {"foo": "bar"}
    assert original_context1 == {"answer": 42}


def test_detector_then_pose_inference_defaults_bbox_scores_when_missing():
    detector_outputs = [
        {
            "bboxes": np.array(
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                ],
                dtype=np.float32,
            )
        }
    ]

    detector_runner = DummyDetectorRunner(detector_outputs)
    pose_runner = RecordingPoseRunner()

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    runner.inference(["img0.png"])

    enriched = pose_runner.calls[0]["images"]
    _, context = enriched[0]

    np.testing.assert_allclose(
        context["bbox_scores"],
        np.array([1.0, 1.0], dtype=np.float32),
    )


def test_detector_then_pose_inference_handles_no_detections():
    detector_outputs = [
        {
            "bboxes": np.zeros((0, 4), dtype=np.float32),
        }
    ]

    detector_runner = DummyDetectorRunner(detector_outputs)
    pose_runner = RecordingPoseRunner()

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    runner.inference(["img0.png"])

    enriched = pose_runner.calls[0]["images"]
    _, context = enriched[0]

    assert isinstance(context["bboxes"], np.ndarray)
    assert isinstance(context["bbox_scores"], np.ndarray)
    assert context["bboxes"].shape == (0, 4)
    assert context["bbox_scores"].shape == (0,)


def test_detector_then_pose_inference_raises_on_output_count_mismatch():
    detector_runner = DummyDetectorRunner(
        [
            {
                "bboxes": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                "bbox_scores": np.array([0.9], dtype=np.float32),
            }
        ]
    )
    pose_runner = RecordingPoseRunner()

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    with pytest.raises(ValueError, match="Detector returned 1 outputs for 2 input images"):
        runner.inference(["img0.png", "img1.png"])

    # Pose runner should not be called if detector output count is invalid
    assert len(pose_runner.calls) == 0


def test_detector_then_pose_inference_raises_on_invalid_bbox_score_length():
    detector_outputs = [
        {
            "bboxes": np.array(
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [50.0, 60.0, 70.0, 80.0],
                ],
                dtype=np.float32,
            ),
            "bbox_scores": np.array([0.5], dtype=np.float32),  # wrong length
        }
    ]

    detector_runner = DummyDetectorRunner(detector_outputs)
    pose_runner = RecordingPoseRunner()

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    with pytest.raises(ValueError, match="Expected one bbox score per bbox"):
        runner.inference(["img0.png"])

    assert len(pose_runner.calls) == 0


def test_detector_then_pose_inference_passes_shelf_writer_through():
    detector_outputs = [
        {
            "bboxes": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
            "bbox_scores": np.array([0.9], dtype=np.float32),
        }
    ]

    detector_runner = DummyDetectorRunner(detector_outputs)
    pose_runner = RecordingPoseRunner()
    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    shelf_writer = object()
    runner.inference(["img0.png"], shelf_writer=shelf_writer)

    assert detector_runner.calls[0]["shelf_writer"] is None
    assert pose_runner.calls[0]["shelf_writer"] is shelf_writer


def test_detector_then_pose_integration_with_real_top_down_preprocessor():
    """
    Integration-style test:
    prove that wrapper-injected context["bboxes"] is consumed by the real top-down
    preprocessor and produces a crop batch of shape [num_individuals, 3, H, W].
    """
    preprocessor = build_top_down_preprocessor(
        color_mode="RGB",
        transform=A.Compose(
            [],
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        ),
        top_down_crop_size=(32, 24),  # width, height
        top_down_crop_margin=0,
        top_down_crop_with_context=True,
    )

    pose_runner = PreprocessingPoseRunner(preprocessor=preprocessor)

    detector_runner = DummyDetectorRunner(
        [
            {
                "bboxes": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [40.0, 15.0, 30.0, 25.0],
                    ],
                    dtype=np.float32,
                ),
                "bbox_scores": np.array([0.8, 0.9], dtype=np.float32),
            }
        ]
    )

    runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
    )

    image = np.zeros((100, 120, 3), dtype=np.uint8)

    outputs = runner.inference([image])

    assert len(outputs) == 1
    out = outputs[0]

    # ToTensor converts NHWC -> NCHW
    assert out["image_shape"] == (2, 3, 24, 32)
    assert out["num_bboxes"] == 2

    # Offsets/scales are produced per crop
    assert out["offsets_shape"] == (2, 2)
    assert out["scales_shape"] == (2, 2)

    # TopDownCrop stores output_size as (width, height)
    assert out["top_down_crop_size"] == (32, 24)


class TinyModel(nn.Module):
    def forward(self, x, **kwargs):
        return x


def test_build_inference_runner_wraps_top_down_runner_when_detector_runner_is_given():
    model = TinyModel()
    detector_runner = DummyDetectorRunner(outputs=[])

    runner = build_inference_runner(
        task=Task.TOP_DOWN,
        model=model,
        device="cpu",
        snapshot_path=None,
        batch_size=1,
        preprocessor=None,
        postprocessor=None,
        detector_runner=detector_runner,
    )

    assert isinstance(runner, DetectorToPoseInferenceRunner)
