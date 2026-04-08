from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import deeplabcut.pose_estimation_pytorch.data.base as base_mod
from deeplabcut.pose_estimation_pytorch import build_training_runner
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.bboxes import BBoxComputationMethod, BBoxEntry, BBoxes
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.models.detectors.external import PrecomputedDetectorRunner
from deeplabcut.pose_estimation_pytorch.task import Task

# -----------------------------------------------------------------------------
# Tiny dataset stand-in so we can inspect create_dataset() output directly
# -----------------------------------------------------------------------------


class DummyPoseDataset:
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


@pytest.fixture(autouse=True)
def patch_pose_dataset(monkeypatch):
    monkeypatch.setattr(base_mod, "PoseDataset", DummyPoseDataset)


# -----------------------------------------------------------------------------
# Fake multi-animal DLC-style loader
# -----------------------------------------------------------------------------


class FakeMultiAnimalDLCLoader(Loader):
    """
    Minimal multi-animal loader:
      - one image
      - two individuals
      - each individual has keypoints that imply a different bbox
    """

    def __init__(self, precomputed_bboxes_path: Path):
        self.project_root = Path(".")
        self.image_root = Path(".")
        self.model_config_path = Path("dummy_pytorch_config.yaml")

        self.model_cfg = {
            "method": "td",
            "data": {
                "bbox_source": BBoxComputationMethod.DETECTION_BBOX.value,
                "precomputed_bboxes": precomputed_bboxes_path.as_posix(),
                "bbox_margin": 5,
                "bbox_match_iou_threshold": 0.1,
                "bbox_fallback_to_gt": True,
            },
            "runner": {},
            "train_settings": {},
        }

        self.pose_task = Task.TOP_DOWN
        self._loaded_data = {}

        # Two individuals in one image, with clearly separated keypoints
        # Individual A (left side)
        keypoints_a = np.array(
            [
                [20.0, 20.0, 2.0],
                [30.0, 30.0, 2.0],
            ],
            dtype=np.float32,
        )

        # Individual B (right side)
        keypoints_b = np.array(
            [
                [70.0, 20.0, 2.0],
                [80.0, 30.0, 2.0],
            ],
            dtype=np.float32,
        )

        self._payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "img0.png",
                    "width": 100,
                    "height": 60,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "individual": "animal_a",
                    "individual_id": 0,
                    # placeholder/stale bbox - should be replaced
                    "bbox": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    "area": 12.0,
                    "keypoints": keypoints_a,
                    "num_keypoints": 2,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "individual": "animal_b",
                    "individual_id": 1,
                    # placeholder/stale bbox - should be replaced
                    "bbox": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                    "area": 56.0,
                    "keypoints": keypoints_b,
                    "num_keypoints": 2,
                    "iscrowd": 0,
                },
            ],
        }

    def load_data(self, mode: str = "train"):
        self._loaded_data.setdefault(mode, self._payload)
        return self._loaded_data[mode]

    def get_dataset_parameters(self) -> PoseDatasetParameters:
        return PoseDatasetParameters(
            bodyparts=["nose", "tail"],
            unique_bpts=[],
            individuals=["animal_a", "animal_b"],
            with_center_keypoints=False,
            color_mode="RGB",
            top_down_crop_size=(64, 64),
            top_down_crop_margin=0,
            top_down_crop_with_context=True,
        )

    def default_bbox_method(self, task: Task):
        # DLCLoader-like backward compatibility
        if task in (Task.TOP_DOWN, Task.DETECT):
            return BBoxComputationMethod.KEYPOINTS
        return None


# -----------------------------------------------------------------------------
# Tiny train dataset for PoseTrainingRunner
# -----------------------------------------------------------------------------


class TinyTrainDataset(Dataset):
    """
    Minimal dataset that yields the batch structure expected by PoseTrainingRunner.

    It uses the annotations produced by create_dataset(...), so training still depends
    on the offline / precomputed detector assignment done earlier.
    """

    def __init__(self, annotations: list[dict]):
        self.annotations = annotations

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # Build keypoints tensor from the matched annotations
        # shape: [num_individuals, num_bodyparts, 3]
        kpts = np.stack([ann["keypoints"] for ann in self.annotations], axis=0).astype(np.float32)

        sample = {
            "image": torch.zeros((3, 32, 32), dtype=torch.float32),
            "annotations": {
                "keypoints": torch.tensor(kpts, dtype=torch.float32),
                "with_center_keypoints": torch.tensor(False),
            },
            "offsets": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "scales": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "context": {},
        }
        return sample


# -----------------------------------------------------------------------------
# Tiny pose model compatible with PoseTrainingRunner
# -----------------------------------------------------------------------------


class TinyPoseModel(nn.Module):
    """
    Minimal trainable pose model:
      - one scalar parameter
      - produces dummy pose predictions
      - implements the methods PoseTrainingRunner expects
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # only needed if someone ever uses load_head_weights=False
        self.backbone = nn.Identity()

    def forward(self, x, cond_kpts=None):
        batch_size = x.shape[0]

        # Predict 2 individuals x 2 bodyparts x 3 values (x, y, visibility)
        pred = torch.ones((batch_size, 2, 2, 3), device=x.device, dtype=torch.float32) * self.weight
        pred[..., 2] = 1.0
        return {"pred_keypoints": pred}

    def get_target(self, outputs, annotations):
        return annotations["keypoints"].to(outputs["pred_keypoints"].device).float()

    def get_loss(self, outputs, target):
        pred_xy = outputs["pred_keypoints"][..., :2]
        target_xy = target[..., :2]
        loss = ((pred_xy - target_xy) ** 2).mean()
        return {"total_loss": loss}

    def get_predictions(self, outputs):
        return {
            "bodypart": {
                "poses": outputs["pred_keypoints"],
            }
        }


# -----------------------------------------------------------------------------
# The actual end-to-end test
# -----------------------------------------------------------------------------


def test_offline_precomputed_topdown_multi_animal_training_e2e(tmp_path: Path):
    """
    End-to-end test for the offline / precomputed external detector workflow.

    Proves that:
      1. precomputed detector boxes can be loaded from config
      2. create_dataset(...) builds the correct multi-animal top-down dataset
      3. training runs through the high-level training API
      4. only the pose model is trained
      5. the detector is not needed anymore once the dataset is built
    """

    # -------------------------------------------------------------------------
    # 1. Create precomputed detector artifact with boxes intentionally reversed
    #    relative to annotation order. Matching must recover the correct assignment.
    # -------------------------------------------------------------------------
    bboxes_path = tmp_path / "precomputed_bboxes.json"

    precomputed = BBoxes(
        train=[
            BBoxEntry(
                # reversed order on purpose:
                # first bbox belongs to animal_b (right side), second to animal_a (left side)
                bboxes=[
                    (65.0, 15.0, 20.0, 20.0),  # should match annotation 2 / animal_b
                    (15.0, 15.0, 20.0, 20.0),  # should match annotation 1 / animal_a
                ],
                bbox_scores=[0.9, 0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
        test=[
            BBoxEntry(
                bboxes=[
                    (65.0, 15.0, 20.0, 20.0),
                    (15.0, 15.0, 20.0, 20.0),
                ],
                bbox_scores=[0.9, 0.8],
                bbox_format="xywh",
                image_path=Path("img0.png"),
            )
        ],
    )
    precomputed.dump_json(bboxes_path)

    # -------------------------------------------------------------------------
    # 2. Build loader + precomputed detector runner from config-like state
    # -------------------------------------------------------------------------
    loader = FakeMultiAnimalDLCLoader(precomputed_bboxes_path=bboxes_path)

    detector_runner = PrecomputedDetectorRunner.from_bboxes(
        BBoxes.from_file(bboxes_path),
        mode="train",
        target_format="xywh",
        validate_image_paths=True,
    )

    # -------------------------------------------------------------------------
    # 3. Create top-down dataset using offline precomputed detector boxes
    # -------------------------------------------------------------------------
    raw_before = [np.asarray(ann["bbox"], dtype=np.float32).copy() for ann in loader.load_data("train")["annotations"]]

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=detector_runner,
    )

    # Annotation order is [animal_a, animal_b].
    # Matching should recover the correct detector box for each animal
    # even though the detector outputs were stored in reversed order.
    actual_bbox_a = np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32)
    actual_bbox_b = np.asarray(dataset.annotations[1]["bbox"], dtype=np.float32)

    expected_bbox_a = np.asarray([15.0, 15.0, 20.0, 20.0], dtype=np.float32)
    expected_bbox_b = np.asarray([65.0, 15.0, 20.0, 20.0], dtype=np.float32)

    np.testing.assert_allclose(actual_bbox_a, expected_bbox_a)
    np.testing.assert_allclose(actual_bbox_b, expected_bbox_b)

    # Cached raw annotations must remain untouched
    raw_after = [np.asarray(ann["bbox"], dtype=np.float32) for ann in loader.load_data("train")["annotations"]]
    np.testing.assert_allclose(raw_before[0], raw_after[0])
    np.testing.assert_allclose(raw_before[1], raw_after[1])

    # -------------------------------------------------------------------------
    # 4. Once dataset is built, training should no longer depend on detector I/O
    #    Prove this by making detector inference crash if called again.
    # -------------------------------------------------------------------------
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("Detector inference should not be called during pose training when using offline data.")

    detector_runner.inference = _should_not_be_called  # type: ignore[method-assign]

    # -------------------------------------------------------------------------
    # 5. Build tiny train/valid loaders from the matched annotations
    # -------------------------------------------------------------------------
    train_ds = TinyTrainDataset(dataset.annotations)
    valid_ds = TinyTrainDataset(dataset.annotations)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)

    # -------------------------------------------------------------------------
    # 6. Build high-level training runner
    # -------------------------------------------------------------------------
    model = TinyPoseModel()

    runner_config = {
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1,
            },
        },
        "eval_interval": 1,
        "snapshots": {
            "max_snapshots": 1,
            "save_epochs": 1,
            "save_optimizer_state": True,
        },
    }

    model_folder = tmp_path / "models"
    model_folder.mkdir(parents=True, exist_ok=True)

    runner = build_training_runner(
        runner_config=runner_config,
        model_folder=model_folder,
        task=Task.TOP_DOWN,
        model=model,
        device="cpu",
        snapshot_path=None,
    )

    # -------------------------------------------------------------------------
    # 7. Assert optimizer only contains trainable pose params
    # -------------------------------------------------------------------------
    optimizer_param_ids = {id(p) for group in runner.optimizer.param_groups for p in group["params"]}
    model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

    assert optimizer_param_ids == model_param_ids

    # -------------------------------------------------------------------------
    # 8. Run one short training cycle and assert pose params changed
    # -------------------------------------------------------------------------
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}

    runner.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=1,
        display_iters=1,
    )

    after = {name: p.detach().cpu() for name, p in model.named_parameters()}

    changed = []
    for name in before:
        if not torch.equal(before[name], after[name]):
            changed.append(name)

    assert len(changed) > 0, "Expected at least one pose model parameter to change during training."
