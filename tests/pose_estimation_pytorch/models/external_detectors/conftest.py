from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import deeplabcut.pose_estimation_pytorch.data.base as base_mod
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.task import Task


class DummyPoseDataset:
    """
    Tiny stand-in for PoseDataset so tests can inspect what create_dataset()
    passes through without depending on the real dataset internals.
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
        # Avoid calling Loader.__init__() because we want a tiny controlled fixture.
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
        if task in (Task.TOP_DOWN, Task.DETECT):
            return "keypoints"
        return None


class RecordingDetectorRunner:
    """
    Tiny detector runner used to test precompute resume/recompute policies.
    """

    def __init__(
        self,
        *,
        bbox: tuple[float, float, float, float] = (91.0, 92.0, 93.0, 94.0),
        score: float = 0.99,
    ):
        self.bbox = bbox
        self.score = score
        self.calls: list[Path] = []

    def inference(self, images, shelf_writer=None, **kwargs):
        images = list(images)
        self.calls.extend([Path(p) for p in images])

        return [
            {
                "bboxes": np.asarray([self.bbox], dtype=np.float32),
                "bbox_scores": np.asarray([self.score], dtype=np.float32),
            }
            for _ in images
        ]


@pytest.fixture(autouse=True)
def patch_pose_dataset(monkeypatch):
    """
    Replace PoseDataset with a tiny dummy object for tests under this directory.
    """
    monkeypatch.setattr(base_mod, "PoseDataset", DummyPoseDataset)


@pytest.fixture
def fake_dlc_loader():
    return FakeDLCLoader()


@pytest.fixture
def fake_snapshot(tmp_path):
    return SimpleNamespace(
        path=tmp_path / "snapshot-40.pt",
        epochs=40,
        uid=lambda: "40",
    )


@pytest.fixture
def eval_mod():
    import deeplabcut.pose_estimation_pytorch.apis.evaluation as evaluation_module

    return evaluation_module


@pytest.fixture
def fake_eval_topdown_loader(tmp_path):
    class FakeEvalLoader:
        pose_task = Task.TOP_DOWN
        train_fraction = 0.5
        shuffle = 4
        evaluation_folder = tmp_path / "eval"

        # evaluate_snapshot(...) reindexes predictions to loader.df.index.
        df = pd.DataFrame(index=["train_img.png", "test_img.png"])

        model_cfg = {
            "method": "td",
            "metadata": {
                "with_identity": False,
                "bodyparts": ["nose", "tail"],
                "unique_bodyparts": [],
                "individuals": ["animal"],
            },
            "model": {
                "heads": {
                    "bodypart": {
                        "type": "HeatmapHead",
                    }
                }
            },
            "runner": {},
            "train_settings": {},
            "data": {
                "bbox_source": "detection_bbox",
                "precomputed_bboxes": str(tmp_path / "precomputed_bboxes.json"),
                "bbox_validate_image_paths": True,
                "bbox_max_detections": 1,
                "bbox_selection_strategy": "score_then_area",
                "bbox_filter_invalid_boxes": True,
            },
        }

        def get_dataset_parameters(self):
            return SimpleNamespace(
                max_num_animals=1,
                num_joints=2,
                num_unique_bpts=0,
                bodyparts=["nose", "tail"],
                unique_bpts=[],
                individuals=["animal"],
            )

    return FakeEvalLoader()


@pytest.fixture
def patch_evaluate_snapshot_dependencies(monkeypatch, eval_mod):
    """
    Patch heavy evaluation dependencies so evaluate_snapshot(...) can run as a unit test.

    Returns state dict with:
      - built_modes
      - evaluate_calls
    """
    state = {
        "built_modes": [],
        "evaluate_calls": [],
    }

    fake_pose_runner = object()
    fake_native_detector_runner = None

    monkeypatch.setattr(
        eval_mod,
        "get_inference_runners",
        lambda **kwargs: (fake_pose_runner, fake_native_detector_runner),
    )

    def fake_build_precomputed_detector_runner_from_config(model_cfg, mode, **kwargs):
        state["built_modes"].append(mode)
        return SimpleNamespace(mode=mode)

    monkeypatch.setattr(
        eval_mod,
        "build_precomputed_detector_runner_from_config",
        fake_build_precomputed_detector_runner_from_config,
    )

    def fake_evaluate(**kwargs):
        state["evaluate_calls"].append(kwargs)

        mode = kwargs["mode"]
        image_name = f"{mode}_img.png"

        return (
            {
                "rmse": 1.0,
                "rmse_pcutoff": 1.0,
                "mAP": 99.0,
                "mAR": 99.0,
            },
            {
                image_name: {
                    "bodyparts": np.zeros((1, 2, 3), dtype=float),
                    "bboxes": np.zeros((1, 4), dtype=float),
                    "bbox_scores": np.ones((1,), dtype=float),
                }
            },
        )

    monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)

    def fake_build_predictions_dataframe(**kwargs):
        predictions = kwargs["predictions"]
        return pd.DataFrame(index=list(predictions.keys()))

    monkeypatch.setattr(
        eval_mod,
        "build_predictions_dataframe",
        fake_build_predictions_dataframe,
    )

    monkeypatch.setattr(
        eval_mod,
        "build_bboxes_dict_for_dataframe",
        lambda **kwargs: {},
    )

    monkeypatch.setattr(
        eval_mod,
        "save_evaluation_results",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_hdf",
        lambda self, *args, **kwargs: None,
    )

    return state


@pytest.fixture
def patch_evaluate_network_dependencies(monkeypatch, eval_mod, tmp_path, fake_snapshot):
    class FakeLoader:
        def __init__(self, *args, **kwargs):
            self.pose_task = Task.TOP_DOWN
            self.model_folder = tmp_path / "train"
            self.model_folder.mkdir(parents=True, exist_ok=True)
            self.evaluation_folder = tmp_path / "eval"
            self.train_fraction = 0.5
            self.model_cfg = {
                "method": "td",
                "device": "cpu",
                "data": {
                    "bbox_source": "detection_bbox",
                    "precomputed_bboxes": str(tmp_path / "precomputed_bboxes.json"),
                    "bbox_validate_image_paths": True,
                    "bbox_max_detections": 1,
                    "bbox_selection_strategy": "score_then_area",
                    "bbox_filter_invalid_boxes": True,
                },
            }

    monkeypatch.setattr(eval_mod, "DLCLoader", FakeLoader)

    monkeypatch.setattr(
        eval_mod.auxiliaryfunctions,
        "read_config",
        lambda path: {
            "TrainingFraction": [0.5],
            "snapshotindex": -1,
            "detector_snapshotindex": -1,
        },
    )

    monkeypatch.setattr(eval_mod.utils, "resolve_device", lambda cfg: "cpu")

    def fake_get_model_snapshots(index, model_folder, task, snapshot_filter=None):
        if task == Task.DETECT:
            return []
        return [fake_snapshot]

    monkeypatch.setattr(eval_mod, "get_model_snapshots", fake_get_model_snapshots)
    monkeypatch.setattr(eval_mod, "get_scorer_uid", lambda snapshot, detector_snapshot: "snapshot_40")
    monkeypatch.setattr(eval_mod, "get_scorer_name", lambda **kwargs: "fake_scorer")
    monkeypatch.setattr(eval_mod, "evaluate_snapshot", lambda **kwargs: None)
