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
"""Tests bbox-source behavior for dataset creation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.data.base as base_mod
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.bboxes import BBoxComputationMethod
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
from deeplabcut.pose_estimation_pytorch.data.utils import bbox_from_keypoints
from deeplabcut.pose_estimation_pytorch.task import Task


class DummyPoseDataset:
    """Tiny stand-in for PoseDataset so we can inspect what create_dataset passes through."""

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
    Minimal Loader used to test create_dataset() logic without needing a real DLC project.
    It mimics DLCLoader's backward-compatible default bbox behavior.
    """

    def __init__(self, bbox_source: str | None = None):
        # Do not call Loader.__init__() — we set just what create_dataset() needs.
        self.project_root = Path(".")
        self.image_root = Path(".")
        self.model_config_path = Path("dummy_pytorch_config.yaml")

        self.model_cfg = {
            "method": "td",
            "data": {
                "bbox_margin": 7,  # IMPORTANT: used to test that configured margin is respected
            },
            "train_settings": {},
        }
        if bbox_source is not None:
            self.model_cfg["data"]["bbox_source"] = bbox_source

        self.pose_task = Task.TOP_DOWN
        self._loaded_data = {}

        # One cached payload, reused across calls — useful to detect accidental mutation
        self._payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "img0.png",
                    "width": 100,
                    "height": 80,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "individual": "animal",
                    "individual_id": 0,
                    # Placeholder bbox that should be replaced in keypoint mode
                    "bbox": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    "area": 12.0,
                    # Two visible keypoints
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
        # Mimic the new DLCLoader backward-compatible behavior
        if task in (Task.TOP_DOWN, Task.DETECT):
            return "keypoints"
        return None


class DummyDetectorRunner:
    """Simple detector runner returning one bbox per image."""

    def __init__(self, bbox, score=0.9):
        self._bbox = np.asarray(bbox, dtype=np.float32)
        self._score = float(score)

    def inference(self, images, shelf_writer=None):
        return [
            {
                "bboxes": np.asarray([self._bbox], dtype=np.float32),
                "bbox_scores": np.asarray([self._score], dtype=np.float32),
            }
            for _ in images
        ]


@pytest.fixture(autouse=True)
def patch_pose_dataset(monkeypatch):
    """
    Replace PoseDataset with a tiny dummy object so tests focus purely on loader logic.
    """
    monkeypatch.setattr(base_mod, "PoseDataset", DummyPoseDataset)


def test_dlcloader_default_bbox_method_is_backward_compatible():
    """
    DLCLoader should preserve historical behavior:
    detector and top-down tasks default to keypoint-derived boxes.
    """
    loader = object.__new__(DLCLoader)

    assert DLCLoader.default_bbox_method(loader, Task.TOP_DOWN) == BBoxComputationMethod.KEYPOINTS
    assert DLCLoader.default_bbox_method(loader, Task.DETECT) == BBoxComputationMethod.KEYPOINTS
    assert DLCLoader.default_bbox_method(loader, Task.BOTTOM_UP) is None


@pytest.mark.parametrize("task", [Task.TOP_DOWN, Task.DETECT])
def test_create_dataset_defaults_to_keypoints_for_dlc_style_loader(task):
    """
    Backward compatibility regression test:
    when no bbox_source is explicitly configured, a DLCLoader-like loader should
    derive boxes from keypoints for TOP_DOWN and DETECT tasks.
    """
    loader = FakeDLCLoader()

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=task,
        detector_runner=None,
    )

    ann = dataset.annotations[0]
    actual_bbox = np.asarray(ann["bbox"], dtype=np.float32)

    expected_bbox = bbox_from_keypoints(
        keypoints=loader._payload["annotations"][0]["keypoints"],
        image_h=loader._payload["images"][0]["height"],
        image_w=loader._payload["images"][0]["width"],
        margin=loader.model_cfg["data"]["bbox_margin"],
    ).astype(np.float32)

    # Ensure configured bbox_margin is respected
    np.testing.assert_allclose(actual_bbox, expected_bbox)

    # Stronger regression guard:
    # this should NOT be the hardcoded margin=20 result from _add_bbox_annotations()
    hardcoded_bbox = bbox_from_keypoints(
        keypoints=loader._payload["annotations"][0]["keypoints"],
        image_h=loader._payload["images"][0]["height"],
        image_w=loader._payload["images"][0]["width"],
        margin=20,
    ).astype(np.float32)

    assert not np.allclose(actual_bbox, hardcoded_bbox), (
        "create_dataset() appears to be relying on the hardcoded bbox=20 fallback "
        "instead of recomputing with configured bbox_margin"
    )


def test_explicit_bbox_source_gt_preserves_existing_bbox():
    """
    Explicit bbox_source='gt' must override the backward-compatible default and keep
    the annotation bbox unchanged.
    """
    loader = FakeDLCLoader(bbox_source="gt")

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=None,
    )

    ann = dataset.annotations[0]
    actual_bbox = np.asarray(ann["bbox"], dtype=np.float32)
    expected_bbox = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    np.testing.assert_allclose(actual_bbox, expected_bbox)


def test_detector_runner_overrides_default_bbox_source():
    """
    If a detector_runner is provided, create_dataset() must use detector boxes even if
    the loader would otherwise default to keypoint-derived boxes.
    """
    loader = FakeDLCLoader()
    detector_runner = DummyDetectorRunner(bbox=[11.0, 12.0, 13.0, 14.0], score=0.95)

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    ann = dataset.annotations[0]
    actual_bbox = np.asarray(ann["bbox"], dtype=np.float32)

    np.testing.assert_allclose(actual_bbox, np.asarray([11.0, 12.0, 13.0, 14.0], dtype=np.float32))


def test_create_dataset_does_not_mutate_cached_load_data_annotations():
    """
    Regression test for the refactor:
    create_dataset() should deep-copy annotations before rewriting bboxes, otherwise
    cached load_data() results become stateful and unsafe across repeated calls.
    """
    loader = FakeDLCLoader()
    detector_runner = DummyDetectorRunner(bbox=[21.0, 22.0, 23.0, 24.0], score=0.88)

    # Sanity check original cached bbox
    raw_before = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32).copy()
    np.testing.assert_allclose(raw_before, np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    # This call should NOT mutate the cached payload
    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    # Dataset bbox should use detector output
    np.testing.assert_allclose(
        np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32),
        np.asarray([21.0, 22.0, 23.0, 24.0], dtype=np.float32),
    )

    # Cached raw annotations must remain untouched
    raw_after = np.asarray(loader.load_data("train")["annotations"][0]["bbox"], dtype=np.float32)
    np.testing.assert_allclose(raw_after, np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_explicit_gt_is_still_overridden_by_detector_runner():
    loader = FakeDLCLoader(bbox_source="gt")
    detector_runner = DummyDetectorRunner(bbox=[31.0, 32.0, 33.0, 34.0])

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    np.testing.assert_allclose(
        np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32),
        np.asarray([31.0, 32.0, 33.0, 34.0], dtype=np.float32),
    )
