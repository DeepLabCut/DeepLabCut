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
"""Tests bbox schema + precomputed detector runner integration with DLC code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.data.base as base_mod
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.bboxes import (
    BBoxEntry,
    BBoxes,
)
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.postprocessor import build_detector_postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import build_bottom_up_preprocessor
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models.detectors.external import EXTERNAL_DETECTORS, PrecomputedDetectorRunner

# Important: ensure the mock detector module is imported so registry population happens
from deeplabcut.pose_estimation_pytorch.runners.inference import build_inference_runner
from deeplabcut.pose_estimation_pytorch.task import Task


class DummyPoseDataset:
    """
    Tiny stand-in for PoseDataset so tests can inspect what create_dataset()
    actually passes through without depending on the real dataset internals.
    """

    def __init__(
        self,
        images,
        annotations,
        transform,
        mode,
        task,
        parameters,
        ctd_config=None,
    ):
        self.images = images
        self.annotations = annotations
        self.transform = transform
        self.mode = mode
        self.task = task
        self.parameters = parameters
        self.ctd_config = ctd_config


class FakeDLCLoader(Loader):
    """
    Minimal loader for testing create_dataset() logic.

    It mimics DLCLoader’s backward-compatible behavior:
    top-down and detect tasks default to keypoint-derived boxes unless a
    detector_runner is provided.
    """

    def __init__(self, bbox_source: str | None = None):
        # Avoid calling Loader.__init__() because we want a tiny controlled fixture
        self.project_root = Path(".")
        self.image_root = Path(".")
        self.model_config_path = Path("dummy_pytorch_config.yaml")
        self.model_cfg = {
            "method": "td",
            "data": {
                "bbox_margin": 7,
            },
            "train_settings": {},
        }
        if bbox_source is not None:
            self.model_cfg["data"]["bbox_source"] = bbox_source

        self.pose_task = Task.TOP_DOWN
        self._loaded_data = {}

        # Cached payload, reused across calls.
        # This lets us test that create_dataset() does NOT mutate cached load_data().
        self._payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "img0.png",
                    "width": 256,
                    "height": 128,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "individual": "animal",
                    "individual_id": 0,
                    # Placeholder bbox that should be overridden
                    "bbox": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    "area": 12.0,
                    "keypoints": np.array(
                        [
                            [30.0, 40.0, 2.0],
                            [50.0, 60.0, 2.0],
                        ],
                        dtype=np.float32,
                    ),
                    "num_keypoints": 2,
                    "iscrowd": 0,
                }
            ],
        }

    def load_data(self, mode: str = "train"):
        self._loaded_data.setdefault(mode, self._payload)
        return self._loaded_data[mode]

    def get_dataset_parameters(self) -> PoseDatasetParameters:
        return PoseDatasetParameters(
            bodyparts=["nose", "tail"],
            unique_bpts=[],
            individuals=["animal"],
            with_center_keypoints=False,
            color_mode="RGB",
            top_down_crop_size=(256, 256),
            top_down_crop_margin=0,
            top_down_crop_with_context=True,
        )

    def default_bbox_method(self, task: Task) -> str | None:
        # Mimic DLCLoader backward compatibility
        if task in (Task.TOP_DOWN, Task.DETECT):
            return "keypoints"
        return None


@pytest.fixture(autouse=True)
def patch_pose_dataset(monkeypatch):
    """
    Replace PoseDataset with a tiny dummy object so tests focus on loader logic.
    """
    monkeypatch.setattr(base_mod, "PoseDataset", DummyPoseDataset)


def test_bbox_entry_from_detector_context_roundtrip_xywh():
    """
    Schema should faithfully round-trip DLC detector context in xywh format.
    """
    context = {
        "bboxes": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        "bbox_scores": np.array([0.9], dtype=np.float32),
    }

    entry = BBoxEntry.from_detector_context(
        context,
        image_path=Path("img0.png"),
        bbox_format="xywh",
    )

    assert entry.image_path == Path("img0.png")
    assert entry.bbox_format == "xywh"
    assert entry.bbox_scores == [0.9]

    restored = entry.to_detector_context(target_format="xywh")
    np.testing.assert_allclose(restored["bboxes"], context["bboxes"])
    np.testing.assert_allclose(restored["bbox_scores"], context["bbox_scores"])


def test_precomputed_detector_runner_inference_matches_dlc_contract():
    """
    PrecomputedDetectorRunner should behave like a DLC detector runner:
    inference(images) -> list[{"bboxes": ..., "bbox_scores": ...}]
    """
    bboxes = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                bbox_scores=[0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    runner = PrecomputedDetectorRunner.from_bboxes(
        bboxes,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    outputs = runner.inference([Path("img0.png")])

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert "bboxes" in outputs[0]
    assert "bbox_scores" in outputs[0]

    np.testing.assert_allclose(
        outputs[0]["bboxes"],
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        outputs[0]["bbox_scores"],
        np.array([0.8], dtype=np.float32),
    )


def test_create_dataset_accepts_precomputed_detector_runner():
    """
    DLC loader.create_dataset(...) should be able to consume PrecomputedDetectorRunner
    and rewrite annotation bboxes accordingly.
    """
    loader = FakeDLCLoader()

    precomputed = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(11.0, 12.0, 13.0, 14.0)],
                bbox_scores=[0.95],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    detector_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    ann = dataset.annotations[0]
    actual_bbox = np.asarray(ann["bbox"], dtype=np.float32)

    np.testing.assert_allclose(
        actual_bbox,
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )


def test_create_dataset_with_precomputed_detector_runner_does_not_mutate_cached_load_data():
    """
    Regression test: create_dataset() must deep-copy cached annotations before rewriting
    bboxes, otherwise load_data() becomes stateful and unsafe.
    """
    loader = FakeDLCLoader()

    precomputed = BBoxes(
        train=[
            BBoxEntry(
                bboxes=[(21.0, 22.0, 23.0, 24.0)],
                bbox_scores=[0.88],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ]
    )

    detector_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    raw_before = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32).copy()
    np.testing.assert_allclose(raw_before, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    np.testing.assert_allclose(
        np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32),
        np.array([21.0, 22.0, 23.0, 24.0], dtype=np.float32),
    )

    # Cached raw annotations must remain untouched
    raw_after = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32)
    np.testing.assert_allclose(raw_after, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_live_mock_detector_can_roundtrip_through_schema_and_precomputed_runner():
    """
    Strong integration test:

      live mock external detector
        -> DLC DetectorInferenceRunner
        -> detector context
        -> BBoxEntry.from_detector_context(...)
        -> PrecomputedDetectorRunner
        -> loader.create_dataset(..., detector_runner=...)

    This proves the schema/adapter layer can bridge live detector outputs
    back into DLC’s training/data path.
    """
    # 1. Build live mock external detector
    detector = EXTERNAL_DETECTORS.build(
        {
            "type": "MockExternalDetector",
            "score": 0.9,
            "label": 1,
        }
    )
    detector.eval()

    # 2. Build DLC detector inference runner around it
    transform = build_transforms({"scale_to_unit_range": True})

    runner = build_inference_runner(
        task=Task.DETECT,
        model=detector,
        device="cpu",
        snapshot_path=None,
        batch_size=1,
        preprocessor=build_bottom_up_preprocessor(
            color_mode="RGB",
            transform=transform,
        ),
        postprocessor=build_detector_postprocessor(
            max_individuals=1,
            min_bbox_score=0.0,
        ),
    )

    # 3. Run detector on a mock image
    image = np.zeros((128, 256, 3), dtype=np.uint8)
    live_outputs = runner.inference([image])

    assert len(live_outputs) == 1
    live_context = live_outputs[0]

    assert "bboxes" in live_context
    assert "bbox_scores" in live_context

    # 4. Convert live DLC detector output -> schema
    entry = BBoxEntry.from_detector_context(
        live_context,
        image_path=Path("img0.png"),
        bbox_format="xywh",  # DLC postprocessed outputs are expected here
    )

    # 5. Build precomputed runner from that schema
    precomputed = BBoxes(train=[entry])

    precomputed_runner = PrecomputedDetectorRunner.from_bboxes(
        precomputed,
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    # 6. Use in DLC create_dataset(...)
    loader = FakeDLCLoader()
    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=precomputed_runner,
    )

    actual_bbox = np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32)
    expected_bbox = np.asarray(live_context["bboxes"][0], dtype=np.float32)

    np.testing.assert_allclose(actual_bbox, expected_bbox)
