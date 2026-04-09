"""
Synthetic end-to-end training demo for the external-detector / precomputed-bbox
workflow in DeepLabCut PyTorch top-down pose estimation.

What this script does
---------------------
1. Creates a minimal, valid DeepLabCut-style project on disk with synthetic data:
   - black RGB frames
   - one white square per frame
   - four annotated keypoints, one at each corner of the square
2. Builds a real ``DLCLoader`` on top of that project.
3. Runs a simple external-style detector runner to generate detector boxes.
4. Saves those boxes via ``precompute_detector_bboxes(...)``.
5. Writes / updates the PyTorch pose config so training uses those precomputed boxes.
6. Verifies that ``DLCLoader.create_dataset(..., detector_runner=...)`` picks up the
   detector boxes before training.
7. Calls the real high-level ``train_network(...)`` API while patching only the pose
   model builder and transforms, keeping the rest of the training workflow canonical.

This file is intended both as:
- a runnable demo script for hackathon participants, and
- a blueprint for an integration test.

Usage
-----
Run as a script:

    python synthetic_square_topdown_train_network_demo.py --output-dir /tmp/dlc_synth_demo

If ``--output-dir`` is omitted, a temporary directory is created automatically.

Notes
-----
- The only intentionally patched parts are the pose-model construction and the
  transform builder. This keeps the focus on the detector-bbox -> DLCLoader ->
  train_network plumbing.
- The detector runner here is deliberately simple: it thresholds the white square
  on a black background and returns the enclosing bbox in ``xywh`` format.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

import deeplabcut.core.config as config_utils
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.apis.training import train_network
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import _yaml_safe_value, make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.data.bboxes import BBoxes
from deeplabcut.pose_estimation_pytorch.data.dlcloader import (
    DLCLoader,
    build_dlc_dataframe_columns,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.external.base import (
    build_precomputed_detector_runner_from_config,
    precompute_detector_bboxes,
)
from deeplabcut.pose_estimation_pytorch.task import Task


class IdentityTopDownTransform:
    """
    Minimal transform object matching the contract expected by PoseDataset.

    It preserves image / keypoints / bboxes exactly as given, and always returns
    a dict containing those keys so dataset.py does not fail on missing 'bboxes'.
    """

    def __call__(self, **kwargs):
        transformed = dict(kwargs)

        # Ensure keys expected downstream always exist
        transformed.setdefault("image", None)
        transformed.setdefault("keypoints", [])
        transformed.setdefault("bboxes", [])

        return transformed

    def __repr__(self):
        return "IdentityTopDownTransform()"


# -----------------------------------------------------------------------------
# Synthetic data helpers
# -----------------------------------------------------------------------------


BODYPARTS = ["tl", "tr", "br", "bl"]
INDIVIDUALS = ["square"]


@dataclass
class SyntheticFrame:
    image: np.ndarray
    bbox_xywh: np.ndarray
    keypoints_xyv: np.ndarray
    rel_index: tuple[str, str, str]
    abs_path: Path


@dataclass
class SyntheticProject:
    project_root: Path
    config_path: Path
    pose_config_path: Path
    precomputed_bboxes_path: Path
    frames: list[SyntheticFrame]


class SquareThresholdDetectorRunner:
    """
    Tiny stand-in for an external detector runner.

    It implements the minimal detector-runner contract:
        inference(images, shelf_writer=None) -> list[{"bboxes": ..., "bbox_scores": ...}]

    The detector simply thresholds non-zero pixels and returns one enclosing bbox per
    image in ``xywh`` format.
    """

    def __init__(self, threshold: int = 1, score: float = 0.99):
        self.threshold = threshold
        self.score = float(score)

    @staticmethod
    def _load_image(item: str | Path | np.ndarray | tuple[Any, dict[str, Any]]) -> np.ndarray:
        if isinstance(item, tuple):
            item = item[0]
        if isinstance(item, np.ndarray):
            return item
        return np.asarray(Image.open(item).convert("RGB"))

    def inference(self, images, shelf_writer=None):
        outputs = []
        for item in images:
            image = self._load_image(item)
            mask = image[..., 0] >= self.threshold
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                bboxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
            else:
                x0 = float(xs.min())
                y0 = float(ys.min())
                x1 = float(xs.max())
                y1 = float(ys.max())
                # inclusive pixel extent -> width/height = max-min+1
                bbox = np.array([[x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]], dtype=np.float32)
                score = np.array([self.score], dtype=np.float32)
                bboxes = bbox
                scores = score

            outputs.append(
                {
                    "bboxes": bboxes,
                    "bbox_scores": scores,
                }
            )
        return outputs


class TinyCornerPoseModel(nn.Module):
    """
    Minimal trainable pose model for one individual with four keypoints.

    This model is deliberately tiny. It is *not* intended as a meaningful production
    architecture; it only serves to make the high-level training path run with a
    lightweight, deterministic model while still exercising:
      - the real DLCLoader
      - the real create_dataset(..., detector_runner=...)
      - the real train_network(...) API
      - the real training runner / optimizer / snapshot machinery
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 12)  # 4 keypoints * (x, y, conf)

    def forward(self, x, cond_kpts=None):
        pooled = self.pool(x).flatten(1)  # [B, 3]
        pred = self.fc(pooled).reshape(len(x), 1, 4, 3)
        pred[..., 2] = torch.sigmoid(pred[..., 2])
        return {"pred_keypoints": pred}

    def get_target(self, outputs, annotations):
        return annotations["keypoints"].float().to(outputs["pred_keypoints"].device)

    def get_loss(self, outputs, target):
        pred = outputs["pred_keypoints"]
        loss_xy = ((pred[..., :2] - target[..., :2]) ** 2).mean()
        loss_conf = ((pred[..., 2] - 1.0) ** 2).mean()
        total = loss_xy + 0.1 * loss_conf
        return {
            "total_loss": total,
            "loss_xy": loss_xy,
            "loss_conf": loss_conf,
        }

    def get_predictions(self, outputs):
        return {
            "bodypart": {
                "poses": outputs["pred_keypoints"],
            }
        }


# -----------------------------------------------------------------------------
# Project construction
# -----------------------------------------------------------------------------


def make_square_image(
    image_size: tuple[int, int] = (128, 128),
    square_xywh: tuple[int, int, int, int] = (32, 40, 24, 24),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create one synthetic RGB frame with a white square on a black background.

    Returns:
        image: uint8 array [H, W, 3]
        bbox_xywh: float32 array [4]
        keypoints_xyv: float32 array [4, 3]
    """
    h, w = image_size
    x, y, bw, bh = square_xywh

    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[y : y + bh, x : x + bw] = 255

    keypoints = np.array(
        [
            [x, y, 2.0],
            [x + bw - 1, y, 2.0],
            [x + bw - 1, y + bh - 1, 2.0],
            [x, y + bh - 1, 2.0],
        ],
        dtype=np.float32,
    )
    bbox = np.array([x, y, bw, bh], dtype=np.float32)
    return image, bbox, keypoints


def _save_rgb_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _create_project_config(project_root: Path) -> tuple[dict, Any]:
    """
    Build a minimal multi-animal DLC project config with a single individual.

    We intentionally use the multi-animal pickle dataset pathway because it is much
    easier to synthesize than the legacy .mat single-animal format.
    """
    cfg_file, yaml_file = af.create_config_template(multianimal=True)
    yaml_file.width = 10_000

    videos_dir = project_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    dummy_video = videos_dir / "dummy.mp4"
    dummy_video.write_bytes(b"")

    cfg_file["Task"] = "synthetic-square"
    cfg_file["scorer"] = "synthetic"
    cfg_file["date"] = "2026-04-09"
    cfg_file["project_path"] = project_root.as_posix()
    cfg_file["video_sets"] = {dummy_video.as_posix(): {"crop": "0, 128, 0, 128"}}

    cfg_file["multianimalproject"] = True
    cfg_file["individuals"] = copy.deepcopy(INDIVIDUALS)
    cfg_file["multianimalbodyparts"] = copy.deepcopy(BODYPARTS)
    cfg_file["uniquebodyparts"] = []
    cfg_file["bodyparts"] = "MULTI!"

    cfg_file["TrainingFraction"] = [0.75]
    cfg_file["iteration"] = 0
    cfg_file["snapshotindex"] = -1

    return cfg_file, yaml_file


def _build_collected_data_dataframe(
    scorer: str,
    frames: list[SyntheticFrame],
) -> pd.DataFrame:
    from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters

    params = PoseDatasetParameters(
        bodyparts=BODYPARTS,
        unique_bpts=[],
        individuals=INDIVIDUALS,
        with_center_keypoints=False,
        color_mode="RGB",
        top_down_crop_size=(32, 32),
        top_down_crop_margin=0,
        top_down_crop_with_context=True,
    )

    columns = build_dlc_dataframe_columns(scorer, params, with_likelihood=False)

    rows = []
    index = []
    for frame in frames:
        xy = frame.keypoints_xyv[:, :2].reshape(1, len(BODYPARTS), 2)
        rows.append(xy.reshape(-1))
        index.append(frame.rel_index)

    df = pd.DataFrame(
        data=np.stack(rows, axis=0),
        index=pd.MultiIndex.from_tuples(index),
        columns=columns,
    )
    return df.sort_index(axis=0)


def _build_dataset_pickle_entries(frames: list[SyntheticFrame]) -> list[dict[str, Any]]:
    entries = []
    for frame in frames:
        joints = np.array(
            [[i, kp[0], kp[1]] for i, kp in enumerate(frame.keypoints_xyv)],
            dtype=np.float32,
        )
        h, w = frame.image.shape[:2]
        entries.append(
            {
                "image": frame.rel_index,
                "size": (3, h, w),
                "joints": {
                    0: joints,
                },
            }
        )
    return entries


def _ensure_loader_get_image_paths() -> None:
    """
    Compatibility shim for versions where precompute_detector_bboxes(...) expects a
    loader.get_image_paths(...) method but Loader only exposes image_filenames(...).
    """
    if not hasattr(DLCLoader, "get_image_paths"):
        DLCLoader.get_image_paths = DLCLoader.image_filenames


def _write_or_update_pose_config(
    project_cfg: dict,
    pose_config_path: Path,
    precomputed_bboxes: str | Path,
    *,
    crop_size: tuple[int, int] = (32, 32),
    epochs: int = 1,
    batch_size: int = 1,
) -> dict:
    """
    Prefer the canonical make_pytorch_pose_config(...) path when available, then patch
    the resulting config to keep the demo lightweight and deterministic.
    """
    pose_config_path.parent.mkdir(parents=True, exist_ok=True)

    pose_cfg = make_pytorch_pose_config(
        project_config=project_cfg,
        pose_config_path=pose_config_path,
        # method=Task.TOP_DOWN,
        net_type="resnet_50",  # only used for metadata here since we patch the model builder later
        top_down=True,
        save=True,
        precomputed_bboxes=precomputed_bboxes,
        bbox_source="detection_bbox",
        external_detector_metadata={
            "name": "SquareThresholdDetectorRunner",
            "kind": "synthetic_demo",
        },
    )

    # Patch the config down to a minimal, fast, CPU-friendly training setup.
    pose_cfg.setdefault("metadata", {})
    pose_cfg["metadata"]["bodyparts"] = copy.deepcopy(BODYPARTS)
    pose_cfg["metadata"]["unique_bodyparts"] = []
    pose_cfg["metadata"]["individuals"] = copy.deepcopy(INDIVIDUALS)

    pose_cfg["method"] = "td"
    pose_cfg["net_type"] = pose_cfg.get("net_type", "synthetic_demo")
    pose_cfg["color_mode"] = "RGB"
    pose_cfg["with_center_keypoints"] = False

    pose_cfg.setdefault("model", {})
    pose_cfg["model"]["type"] = "TinyCornerPoseModel"

    pose_cfg.setdefault("data", {})
    pose_cfg["data"]["bbox_source"] = "detection_bbox"
    pose_cfg["data"]["precomputed_bboxes"] = Path(precomputed_bboxes).as_posix()
    pose_cfg["data"]["bbox_validate_image_paths"] = False
    pose_cfg["data"].setdefault("bbox_match_iou_threshold", 0.1)
    pose_cfg["data"].setdefault("bbox_fallback_to_gt", True)
    pose_cfg["data"].setdefault("bbox_margin", 0)
    pose_cfg["data"].setdefault("train", {})
    pose_cfg["data"].setdefault("inference", {})
    pose_cfg["data"]["train"].setdefault("top_down_crop", {})
    pose_cfg["data"]["train"]["top_down_crop"].update(
        {
            "width": int(crop_size[0]),
            "height": int(crop_size[1]),
            "margin": 0,
            "crop_with_context": True,
        }
    )
    pose_cfg["data"]["inference"].setdefault("top_down_crop", {})
    pose_cfg["data"]["inference"]["top_down_crop"].update(
        {
            "width": int(crop_size[0]),
            "height": int(crop_size[1]),
            "margin": 0,
            "crop_with_context": True,
        }
    )

    pose_cfg.setdefault("train_settings", {})
    pose_cfg["train_settings"].update(
        {
            "seed": 0,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "dataloader_workers": 0,
            "dataloader_pin_memory": False,
            "display_iters": 1,
        }
    )

    pose_cfg.setdefault("runner", {})
    pose_cfg["runner"]["optimizer"] = {
        "type": "SGD",
        "params": {"lr": 0.1},
    }
    pose_cfg["runner"]["eval_interval"] = 999  # keep demo focused on training path
    pose_cfg["runner"]["snapshots"] = {
        "max_snapshots": 1,
        "save_epochs": 1,
        "save_optimizer_state": True,
    }

    pose_cfg.setdefault("detector", {})
    pose_cfg["detector"].setdefault("train_settings", {})
    pose_cfg["detector"]["train_settings"]["epochs"] = 0

    pose_cfg = _yaml_safe_value(pose_cfg)
    config_utils.write_config(pose_config_path, pose_cfg, overwrite=True)
    return pose_cfg


def make_synthetic_square_dlc_project(
    output_dir: str | Path,
    *,
    num_frames: int = 4,
    image_size: tuple[int, int] = (128, 128),
    crop_size: tuple[int, int] = (32, 32),
    shuffle: int = 1,
) -> SyntheticProject:
    """
    Create a minimal, valid DLC-style project on disk using real image files,
    CollectedData.h5, dataset split pickle, dataset pickle and a PyTorch pose config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) project config
    project_cfg, yaml_file = _create_project_config(output_dir)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml_file.dump(project_cfg, f)

    # 2) synthetic frames on disk
    output_dir / "labeled-data" / "synthetic-square"
    frames: list[SyntheticFrame] = []

    placements = [
        (24, 24, 20, 20),
        (64, 16, 24, 24),
        (20, 72, 28, 16),
        (72, 72, 18, 26),
    ]
    placements = placements[:num_frames]

    for i, square in enumerate(placements):
        image, bbox, keypoints = make_square_image(image_size=image_size, square_xywh=square)
        rel_index = ("labeled-data", "synthetic-square", f"img{i:03d}.png")
        abs_path = output_dir.joinpath(*rel_index)
        _save_rgb_png(image, abs_path)
        frames.append(
            SyntheticFrame(
                image=image,
                bbox_xywh=bbox,
                keypoints_xyv=keypoints,
                rel_index=rel_index,
                abs_path=abs_path,
            )
        )

    # 3) CollectedData_<scorer>.h5
    trainset_dir = output_dir / af.get_training_set_folder(project_cfg)
    trainset_dir.mkdir(parents=True, exist_ok=True)
    collected_path = trainset_dir / f"CollectedData_{project_cfg['scorer']}.h5"
    collected_df = _build_collected_data_dataframe(project_cfg["scorer"], frames)
    collected_df.to_hdf(collected_path, key="df_with_missing")

    # 4) DLC multi-animal dataset pickle
    train_frac = int(100 * project_cfg["TrainingFraction"][0])
    dataset_prefix = f"{project_cfg['Task']}_{project_cfg['scorer']}{train_frac}shuffle{shuffle}"
    dataset_pickle_path = trainset_dir / f"{dataset_prefix}.pickle"
    with open(dataset_pickle_path, "wb") as f:
        pickle.dump(_build_dataset_pickle_entries(frames), f)

    # 5) split pickle consumed by DLCLoader.load_split(...)
    # meta[1] -> train ids, meta[2] -> test ids
    train_ids = list(range(max(1, len(frames) - 1)))
    test_ids = [len(frames) - 1]
    split_pickle_path = trainset_dir / f"Documentation_data-{project_cfg['Task']}_{train_frac}shuffle{shuffle}.pickle"
    with open(split_pickle_path, "wb") as f:
        pickle.dump((None, train_ids, test_ids), f)

    # 6) model folder / PyTorch config path
    model_folder = af.get_model_folder(
        project_cfg["TrainingFraction"][0],
        shuffle,
        project_cfg,
        engine=Engine.PYTORCH,
        modelprefix="",
    )
    pose_config_path = output_dir / model_folder / "train" / Engine.PYTORCH.pose_cfg_name
    precomputed_bboxes_path = output_dir / model_folder / "train" / "precomputed_bboxes.json"

    _write_or_update_pose_config(
        project_cfg=project_cfg,
        pose_config_path=pose_config_path,
        precomputed_bboxes=precomputed_bboxes_path,
        crop_size=crop_size,
        epochs=1,
        batch_size=1,
    )

    return SyntheticProject(
        project_root=output_dir,
        config_path=config_path,
        pose_config_path=pose_config_path,
        precomputed_bboxes_path=precomputed_bboxes_path,
        frames=frames,
    )


# -----------------------------------------------------------------------------
# Workflow helpers
# -----------------------------------------------------------------------------


def generate_precomputed_detector_boxes(project: SyntheticProject, shuffle: int = 1) -> BBoxes:
    """
    Canonical external-detector workflow step:
      1. build a real DLCLoader on the project
      2. run a detector runner
      3. save the results as a BBoxes JSON artifact
    """
    _ensure_loader_get_image_paths()

    loader = DLCLoader(config=project.config_path, shuffle=shuffle, trainset_index=0)
    detector_runner = SquareThresholdDetectorRunner()

    bboxes = precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector_runner,
        output_file=project.precomputed_bboxes_path,
        modes=("train", "test"),
        bbox_format="xywh",
    )
    return bboxes


def verify_loader_uses_precomputed_boxes(project: SyntheticProject, shuffle: int = 1) -> None:
    """
    Pre-flight check before training:
    prove that the real DLCLoader picks up the saved precomputed detector boxes and
    rewrites top-down annotation bboxes accordingly.
    """
    loader = DLCLoader(config=project.config_path, shuffle=shuffle, trainset_index=0)
    runner = build_precomputed_detector_runner_from_config(
        loader.model_cfg,
        mode="train",
        target_format="xywh",
        validate_image_paths=False,
    )
    if runner is None:
        raise RuntimeError("Failed to build a precomputed detector runner from the pose config.")

    dataset = loader.create_dataset(
        transform=None,
        mode="train",
        task=Task.TOP_DOWN,
        detector_runner=runner,
    )

    # Check the first training-frame annotation bbox against the known synthetic square.
    expected = np.asarray(project.frames[0].bbox_xywh, dtype=np.float32)
    found = np.asarray(dataset.annotations[0]["bbox"], dtype=np.float32)
    np.testing.assert_allclose(found, expected, atol=1e-5)


def run_train_network_demo(project: SyntheticProject, shuffle: int = 1) -> TinyCornerPoseModel:
    """
    Run the actual high-level train_network(...) API while patching only:
      - PoseModel.build(...) -> tiny trainable demo model
      - build_transforms(...) -> None (to keep the demo focused on bbox/crop plumbing)

    Returns the trained tiny model instance so callers can inspect parameter changes.
    """
    import deeplabcut.pose_estimation_pytorch.apis.training as training_api

    tiny_model = TinyCornerPoseModel()
    before = {name: p.detach().cpu().clone() for name, p in tiny_model.named_parameters()}

    with (
        patch.object(
            training_api.PoseModel,
            "build",
            side_effect=lambda *args, **kwargs: tiny_model,
        ),
        patch.object(
            training_api,
            "build_transforms",
            side_effect=lambda cfg: IdentityTopDownTransform(),
        ),
    ):
        train_network(
            config=project.config_path,
            shuffle=shuffle,
            trainingsetindex=0,
            device="cpu",
        )

    changed = [name for name, p in tiny_model.named_parameters() if not torch.equal(before[name], p.detach().cpu())]
    if len(changed) == 0:
        raise AssertionError("Expected at least one model parameter to change during train_network(...).")

    return tiny_model


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main(output_dir: str | Path | None = None) -> SyntheticProject:
    owns_tmp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="dlc_synth_square_demo_"))
        owns_tmp = True
    else:
        output_dir = Path(output_dir)

    project = make_synthetic_square_dlc_project(output_dir)
    print(f"[1/4] Synthetic DLC project created at: {project.project_root}")
    print(f"       config.yaml:        {project.config_path}")
    print(f"       pytorch_config.yaml:{project.pose_config_path}")

    bboxes = generate_precomputed_detector_boxes(project)
    print(f"[2/4] Precomputed detector boxes written to: {project.precomputed_bboxes_path}")
    print(f"       train entries: {len(bboxes.train)}, test entries: {len(bboxes.test)}")

    verify_loader_uses_precomputed_boxes(project)
    print("[3/4] Verified: real DLCLoader.create_dataset(...) uses the saved detector boxes.")

    model = run_train_network_demo(project)
    print("[4/4] train_network(...) completed successfully using the real DLCLoader and precomputed detector boxes.")
    print(f"       tiny model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if owns_tmp:
        print("\nNote: a temporary project directory was created automatically.")
        print(f"      You can inspect it here: {project.project_root}")

    return project


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic DLC top-down training demo with precomputed detector boxes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory in which to create the synthetic project. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()
    main(args.output_dir)
