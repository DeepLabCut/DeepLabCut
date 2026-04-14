"""
External-detector workflow example for DeepLabCut PyTorch top-down pose estimation.

This example is intended for those who already have a *real* DeepLabCut project
with labeled data and a created shuffle / PyTorch model folder.

Description
-----------------------------
1. Open a normal DLC project with a real ``config.yaml``.
2. Choose a DLC pose model.
3. Plug in your own external detector by implementing a tiny adapter class.
4. Run the detector offline on the train/test images and save the results as
   ``precomputed_bboxes.json``.
5. Create/update the project's ``pytorch_config.yaml`` so the pose model trains in
   top-down mode using those precomputed boxes.
6. Train the DLC pose model via the ``train_network(...)`` API.
7. Run inference either on:
   - a video (using per-frame bbox context, optionally cached to disk), or
   - a folder of image frames.

Purpose
-----------------------
The goal is to make it easy to use *your own detector* while keeping *DLC pose models*
for training and inference.
In this workflow, the detector is responsible only for
providing bounding boxes (proposals / crops), and DeepLabCut still handles:
- dataset loading,
- crop generation,
- pose-model training,
- snapshot management,
- and inference.

Important prerequisites
-----------------------
Before using this script, you should already have:
1. a normal DeepLabCut project with labeled data (from RCP),
2. a created training dataset / shuffle for the PyTorch engine (provided or your own),
3. and a valid ``config.yaml``.


What you should edit
--------------------
Users should mainly edit:
- ``CONFIG``                -> path to their DLC ``config.yaml``
- ``POSE_MODEL``            -> which DLC pose model to use
- ``MyExternalDetector``    -> the detector adapter, where most of the work will happen
- a few curated training / crop settings in ``USER_SETTINGS``

What you usually should *not* edit (unless you want/have to)
----------------------------------
- ``DLCLoader`` internals
- bbox artifact schema internals
- runner construction internals
- raw snapshot loading
- pose-model internals

Example usage
(CLI if needed, but I'd suggest using a notebook for dev and debug.
RCP makes this easy, just import the script from your notebook and use the functions directly):
-------------
Train only:

    python external_detector_real_project_workflow.py --config /path/to/config.yaml --train

Train + video inference:

    python external_detector_real_project_workflow.py \
        --config /path/to/config.yaml \
        --train \
        --video /path/to/video.mp4

Folder-of-frames inference:

    python external_detector_real_project_workflow.py \
        --config /path/to/config.yaml \
        --images-dir /path/to/frames
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deeplabcut.pose_estimation_pytorch.apis.training import train_network
from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    VideoIterator,
    create_df_from_prediction,
    video_inference,
)
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.models.detectors.external.base import (
    precompute_detector_bboxes,
)
from deeplabcut.pose_estimation_pytorch.runners.inference import DetectorToPoseInferenceRunner
from deeplabcut.pose_estimation_pytorch.task import Task

# -----------------------------------------------------------------------------
# User-facing settings
# -----------------------------------------------------------------------------

EXAMPLE_POSE_MODELS = [
    "hrnet_w32",
    "resnet_50",
    "rtmpose_x",
    "rtmpose_s",
    "rtmpose_m",
]


@dataclass
class UserSettings:
    pose_model: str = "resnet_50"
    shuffle: int = 1
    trainingsetindex: int = 0
    batch_size: int = 4
    epochs: int = 50
    crop_width: int = 256
    crop_height: int = 256
    bbox_match_iou_threshold: float = 0.1
    bbox_fallback_to_gt: bool = True
    bbox_validate_image_paths: bool = False
    display_iters: int = 50
    device: str | None = None


# -----------------------------------------------------------------------------
# Detector adapter section (participants should replace this with their own detector)
# -----------------------------------------------------------------------------


class PretrainedDetectorModel:
    """
    Replace the internals of this class with your own detector.

    Required contract:
        inference(images, shelf_writer=None) -> list[dict]

    For each input image, return a dict in DLC detector-context format:
        {
            "bboxes": np.ndarray[N, 4],      # XYWH in pixels
            "bbox_scores": np.ndarray[N],
        }

    Supported input image elements typically include:
      - ``Path`` / ``str`` to an image file,
      - ``np.ndarray`` image arrays,
      - or ``(image, context)`` tuples.

    The simplest way to adapt your detector is:
      1. load the image if needed,
      2. run your detector,
      3. convert its output boxes to XYWH pixel coordinates,
      4. return the list of per-image dicts.

    Notes
    -----
    - Boxes must be in ``xywh`` format because the current DLC top-down crop path
      expects that downstream.
    - If your detector naturally returns ``xyxy`` boxes, convert them before returning.
      The pose_estimation_pytorch.data.bboxes.BBoxEntry schemas
      already have converter functions in place, feel free to extend them.
    """

    def inference(self, images, shelf_writer=None):
        raise NotImplementedError(
            "Replace `MyExternalDetector.inference(...)` with your own detector adapter.\n"
            "It must return a list of dicts with keys `bboxes` and `bbox_scores`, where\n"
            "`bboxes` has shape [N, 4] in XYWH pixel coordinates."
        )


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------


def ensure_loader_get_image_paths() -> None:
    """
    Compatibility shim for versions where precompute_detector_bboxes(...) expects a
    loader.get_image_paths(...) method but Loader only exposes image_filenames(...).
    """
    if not hasattr(DLCLoader, "get_image_paths"):
        DLCLoader.get_image_paths = DLCLoader.image_filenames


def list_images_in_folder(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    if len(paths) == 0:
        raise FileNotFoundError(f"No supported image files found in {images_dir}")
    return paths


def infer_top_down_flag(pose_model_name: str) -> bool:
    """
    Heuristic for the config builder.

    - Backbone names need `top_down=True` to become TD pose models.
    - Explicit top-down configs like `resnet_50` are already TD, but passing
      `top_down=True` is harmless for the config-builder path.
    - This example is specifically for top-down detector-driven workflows.
    """
    return True


# -----------------------------------------------------------------------------
# Config preparation helpers
# -----------------------------------------------------------------------------


def prepare_external_topdown_pose_config(
    config: str | Path,
    settings: UserSettings,
    precomputed_bboxes_path: str | Path,
    external_detector_metadata: dict | None = None,
    modelprefix: str = "",
) -> tuple[DLCLoader, Path]:
    """
    Create/update the DLC PyTorch pose config for the external / precomputed detector workflow.

    This function:
    1. loads the real DLC project through DLCLoader,
    2. creates / overwrites the project's pytorch_config.yaml using make_pytorch_pose_config(...),
    3. applies a few curated updates relevant to this workflow.
    """
    loader = DLCLoader(
        config=config,
        trainset_index=settings.trainingsetindex,
        shuffle=settings.shuffle,
        modelprefix=modelprefix,
    )

    pose_cfg = make_pytorch_pose_config(
        project_config=loader.project_cfg,
        pose_config_path=loader.model_config_path,
        net_type=settings.pose_model,
        top_down=infer_top_down_flag(settings.pose_model),
        detector_mode="external",
        save=True,
        precomputed_bboxes=precomputed_bboxes_path,
        bbox_source="detection_bbox",
        external_detector_metadata=external_detector_metadata or {},
    )

    # Validate the chosen model really resolves to top-down.
    if Task(pose_cfg["method"]) != Task.TOP_DOWN:
        raise ValueError(
            f"The selected pose model '{settings.pose_model}' did not resolve to a top-down model. "
            f"Choose a top-down-capable model. Recommended examples: {EXAMPLE_POSE_MODELS}"
        )

    # Apply curated configuration updates via the canonical loader.update_model_cfg(...) path.
    cfg_updates = {
        "data.precomputed_bboxes": Path(precomputed_bboxes_path).as_posix(),
        "data.bbox_source": "detection_bbox",
        "data.bbox_match_iou_threshold": settings.bbox_match_iou_threshold,
        "data.bbox_fallback_to_gt": settings.bbox_fallback_to_gt,
        "data.bbox_validate_image_paths": settings.bbox_validate_image_paths,
        "data.train.top_down_crop.width": settings.crop_width,
        "data.train.top_down_crop.height": settings.crop_height,
        "data.inference.top_down_crop.width": settings.crop_width,
        "data.inference.top_down_crop.height": settings.crop_height,
        "train_settings.batch_size": settings.batch_size,
        "train_settings.epochs": settings.epochs,
        "train_settings.display_iters": settings.display_iters,
        # detector training is disabled in the external/offline workflow
        "detector.train_settings.epochs": 0,
    }

    if settings.device is not None:
        cfg_updates["device"] = settings.device

    loader.update_model_cfg(cfg_updates)
    return loader, loader.model_config_path


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def save_external_detector_bboxes(
    config: str | Path,
    detector_runner,
    settings: UserSettings,
    output_file: str | Path,
    modelprefix: str = "",
):
    """
    Run the external detector on the train/test images of a real DLC project and save
    the results as a reusable JSON bbox artifact.
    """
    ensure_loader_get_image_paths()
    loader = DLCLoader(
        config=config,
        trainset_index=settings.trainingsetindex,
        shuffle=settings.shuffle,
        modelprefix=modelprefix,
    )

    return precompute_detector_bboxes(
        loader=loader,
        detector_runner=detector_runner,
        output_file=output_file,
        modes=("train", "test"),
        bbox_format="xywh",
    )


def train_external_topdown_pose_model(
    config: str | Path,
    settings: UserSettings,
    modelprefix: str = "",
) -> None:
    """
    Train the configured top-down pose model using the real DLC train_network(...) API.
    """
    train_network(
        config=config,
        shuffle=settings.shuffle,
        trainingsetindex=settings.trainingsetindex,
        modelprefix=modelprefix,
        device=settings.device,
        batch_size=settings.batch_size,
        epochs=settings.epochs,
        display_iters=settings.display_iters,
    )


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------


def _load_or_compute_video_box_context(
    video_path: str | Path,
    detector_runner,
    cache_file: str | Path | None = None,
) -> list[dict[str, np.ndarray]]:
    """
    Compute (or load) per-frame detector boxes for a video.

    If `cache_file` is provided and exists, contexts are loaded from it.
    Otherwise, the detector is run on the video frames and the result is optionally
    saved to `cache_file`.

    The cache is intentionally a simple pickle so participants can inspect / curate it.
    """
    if cache_file is not None:
        cache_file = Path(cache_file)
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    video_iter = VideoIterator(video_path)
    contexts = detector_runner.inference(list(video_iter))

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(contexts, f, pickle.HIGHEST_PROTOCOL)

    return contexts


def analyze_video_with_external_boxes(
    config: str | Path,
    video: str | Path,
    detector_runner,
    settings: UserSettings,
    modelprefix: str = "",
    video_box_cache: str | Path | None = None,
):
    """
    Run top-down video inference using offline / precomputed per-frame bbox context.

    This uses the current cleanest inference path for external detectors:
      1. compute or load per-frame bbox contexts,
      2. attach them to a VideoIterator,
      3. call video_inference(...) with detector_runner=None.
    """
    loader = DLCLoader(
        config=config,
        trainset_index=settings.trainingsetindex,
        shuffle=settings.shuffle,
        modelprefix=modelprefix,
    )

    snapshots = loader.snapshots(detector=False, best_in_last=True)
    if len(snapshots) == 0:
        raise RuntimeError("No pose snapshots were found. Train the model first.")
    snapshot = snapshots[-1]

    pose_runner = get_pose_inference_runner(
        model_config=loader.model_cfg,
        snapshot_path=snapshot.path,
        batch_size=1,
        device=settings.device,
        max_individuals=len(loader.model_cfg["metadata"]["individuals"]),
        transform=None,
        dynamic=None,
        cond_provider=None,
        ctd_tracking=False,
        inference_cfg=None,
    )

    contexts = _load_or_compute_video_box_context(video, detector_runner, cache_file=video_box_cache)
    video_iterator = VideoIterator(video)
    video_iterator.set_context(contexts)

    predictions = video_inference(
        video=video_iterator,
        pose_runner=pose_runner,
        detector_runner=None,
        cropping=None,
        shelf_writer=None,
        robust_nframes=False,
        show_gpu_memory=False,
    )

    dlc_scorer = loader.scorer(snapshot)
    output_path = Path(video).parent
    output_prefix = Path(video).stem + dlc_scorer + "_external"

    create_df_from_prediction(
        predictions=predictions,
        dlc_scorer=dlc_scorer,
        multi_animal=loader.project_cfg["multianimalproject"],
        model_cfg=loader.model_cfg,
        output_path=output_path,
        output_prefix=output_prefix,
        save_as_csv=False,
    )

    return predictions


def analyze_image_folder_with_external_boxes(
    config: str | Path,
    images_dir: str | Path,
    detector_runner,
    settings: UserSettings,
    modelprefix: str = "",
):
    """
    Run top-down inference on a folder of image frames.

    This uses the precomputed bbox context path directly by building a list of
    `(image_path, context)` tuples and giving them to the pose runner.
    """
    loader = DLCLoader(
        config=config,
        trainset_index=settings.trainingsetindex,
        shuffle=settings.shuffle,
        modelprefix=modelprefix,
    )

    snapshots = loader.snapshots(detector=False, best_in_last=True)
    if len(snapshots) == 0:
        raise RuntimeError("No pose snapshots were found. Train the model first.")
    snapshot = snapshots[-1]

    pose_runner = get_pose_inference_runner(
        model_config=loader.model_cfg,
        snapshot_path=snapshot.path,
        batch_size=1,
        device=settings.device,
        max_individuals=len(loader.model_cfg["metadata"]["individuals"]),
        transform=None,
        dynamic=None,
        cond_provider=None,
        ctd_tracking=False,
        inference_cfg=None,
    )

    image_paths = list_images_in_folder(images_dir)

    composite_runner = DetectorToPoseInferenceRunner(
        pose_runner=pose_runner,
        detector_runner=detector_runner,
        max_individuals=len(loader.model_cfg["metadata"]["individuals"]),
        num_joints=len(loader.model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(loader.model_cfg["metadata"].get("unique_bodyparts", [])),
    )

    predictions = composite_runner.inference(image_paths)

    dlc_scorer = loader.scorer(snapshot)
    output_path = Path(images_dir)
    output_prefix = output_path.name + dlc_scorer + "_external"

    create_df_from_prediction(
        predictions=predictions,
        dlc_scorer=dlc_scorer,
        multi_animal=loader.project_cfg["multianimalproject"],
        model_cfg=loader.model_cfg,
        output_path=output_path,
        output_prefix=output_prefix,
        save_as_csv=True,
    )

    return predictions


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main(
    config: str | Path,
    settings: UserSettings,
    train: bool = False,
    video: str | Path | None = None,
    images_dir: str | Path | None = None,
    modelprefix: str = "",
    video_box_cache: str | Path | None = None,
):
    config = Path(config)
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    # Update the detector args
    detector = PretrainedDetectorModel(...)

    # Build loader once to resolve the canonical model folder.
    loader = DLCLoader(
        config=config,
        trainset_index=settings.trainingsetindex,
        shuffle=settings.shuffle,
        modelprefix=modelprefix,
    )

    bbox_file = loader.model_folder / "precomputed_bboxes.json"

    print("=== External detector + DLC top-down workflow ===")
    print(f"Project config:       {config}")
    print(f"Shuffle:              {settings.shuffle}")
    print(f"Training set index:   {settings.trainingsetindex}")
    print(f"Pose model:           {settings.pose_model}")
    print(f"BBox artifact:        {bbox_file}")
    print()

    print("[1/4] Running external detector on the project images and saving offline boxes...")
    save_external_detector_bboxes(
        config=config,
        detector_runner=detector,
        settings=settings,
        output_file=bbox_file,
        modelprefix=modelprefix,
    )
    print("        Done.")

    print("[2/4] Creating/updating pytorch_config.yaml for external top-down training...")
    loader, pose_cfg_path = prepare_external_topdown_pose_config(
        config=config,
        settings=settings,
        precomputed_bboxes_path=bbox_file,
        external_detector_metadata={
            "name": detector.__class__.__name__,
            "integration": "external_offline_boxes_example",
        },
        modelprefix=modelprefix,
    )
    print(f"        Wrote pose config: {pose_cfg_path}")

    if train:
        print("[3/4] Training the DLC pose model with offline detector boxes...")
        train_external_topdown_pose_model(config=config, settings=settings, modelprefix=modelprefix)
        print("        Training finished.")
    else:
        print("[3/4] Skipping training (--train not given).")

    if video is not None and images_dir is not None:
        raise ValueError("Please provide either --video or --images-dir, not both.")

    if video is not None:
        print("[4/4] Running video inference with offline boxes...")
        preds = analyze_video_with_external_boxes(
            config=config,
            video=video,
            detector_runner=detector,
            settings=settings,
            modelprefix=modelprefix,
            video_box_cache=video_box_cache,
        )
        print(f"        Wrote predictions for {len(preds)} video frames.")
    elif images_dir is not None:
        print("[4/4] Running image-folder inference with offline boxes...")
        preds = analyze_image_folder_with_external_boxes(
            config=config,
            images_dir=images_dir,
            detector_runner=detector,
            settings=settings,
            modelprefix=modelprefix,
        )
        print(f"        Wrote predictions for {len(preds)} images.")
    else:
        print("[4/4] No inference target provided. Use --video or --images-dir to run inference.")

    print()
    print("Workflow complete.")
    print("Benchmark time :3")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External detector + DLC top-down workflow example (offline boxes).")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the DLC project config.yaml",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="top_down_resnet_50",
        help=(f"DLC pose model to use. You can pass a raw DLC net_type. Recommended examples: {EXAMPLE_POSE_MODELS}"),
    )
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index")
    parser.add_argument("--trainingsetindex", type=int, default=0, help="TrainingFraction index")
    parser.add_argument("--batch-size", type=int, default=4, help="Pose training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Pose training epochs")
    parser.add_argument("--crop-width", type=int, default=256, help="Top-down crop width")
    parser.add_argument("--crop-height", type=int, default=256, help="Top-down crop height")
    parser.add_argument("--display-iters", type=int, default=50, help="Loss logging interval during training")
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu/cuda/mps")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training after preparing the offline bbox artifact and pose config.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video on which to run inference using offline boxes.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional path to a folder of image frames on which to run inference using offline boxes.",
    )
    parser.add_argument(
        "--video-box-cache",
        type=str,
        default=None,
        help="Optional pickle cache for per-frame video detector boxes.",
    )
    parser.add_argument(
        "--modelprefix",
        type=str,
        default="",
        help="Optional DLC modelprefix if your project uses one.",
    )

    args = parser.parse_args()

    settings = UserSettings(
        pose_model=args.pose_model,
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        batch_size=args.batch_size,
        epochs=args.epochs,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        display_iters=args.display_iters,
        device=args.device,
    )

    main(
        config=args.config,
        settings=settings,
        train=args.train,
        video=args.video,
        images_dir=args.images_dir,
        modelprefix=args.modelprefix,
        video_box_cache=args.video_box_cache,
    )
