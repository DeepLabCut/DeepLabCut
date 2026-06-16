from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
import torch
import torch.nn as nn

# from tqdm.auto import tqdm
from PIL import Image

from deeplabcut.pose_estimation_pytorch.data.base import DetectorRunnerLike, Loader
from deeplabcut.pose_estimation_pytorch.data.bboxes import (
    BBoxEntry,
    BBoxes,
    BBoxFormat,
    DetectorContext,
    EvalMode,
    _xyxy_to_xywh,
)
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


class DetectionResult(TypedDict, total=False):
    boxes: torch.Tensor  # FloatTensor [N, 4], absolute XYXY pixel coords
    scores: torch.Tensor  # FloatTensor [N]
    labels: torch.Tensor  # LongTensor [N]
    # Optional future extensions:
    # masks: torch.Tensor
    # embeddings: torch.Tensor
    # class_names: list[str]


class DetectorForwardLike(Protocol):
    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]: ...


def _build_external_detector(
    cfg: dict,
    **kwargs,
) -> BaseExternalDetector:
    """
    Builds an external detector from config.

    Unlike native DLC detectors, external detectors are assumed to be
    inference-oriented and usually are not trained (but the pose estimation model may be trained on top of them).
    As such, external detectors are not expected to have a training loop, and may not even have an optimizer or
    snapshot loading, or target generation.
    """
    detector: BaseExternalDetector = build_from_cfg(cfg, **kwargs)
    return detector


EXTERNAL_DETECTORS = Registry("external_detectors", build_func=_build_external_detector)


class BaseExternalDetector(ABC, nn.Module):
    """
    Base class for external / frozen detectors.

    Detector subclasses implement:
        predict(images) -> list[DetectionResult]

    The base class provides:
        forward(...)
            for nn.Module / runner compatibility

        inference(...)
            for DLC DetectorRunnerLike compatibility

    Therefore external detectors can be passed directly anywhere DLC expects
    a DetectorRunnerLike.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(
        self,
        images: list[torch.Tensor],
    ) -> list[DetectionResult]:
        """
        Run detection on a batch of images.

        Args:
            images:
                List of images, each typically a tensor of shape [C, H, W].

        Returns:
            One detection dict per image:
                {
                    "boxes": FloatTensor[N, 4],   # XYXY absolute pixel coords
                    "scores": FloatTensor[N],
                    "labels": LongTensor[N],
                }
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[DetectionResult]]:
        """
        Compatibility shim so external detectors can be used with existing
        inference-runner code that expects nn.Module.forward().

        For inference-only external detectors, losses are always empty.
        """
        if isinstance(x, torch.Tensor):
            images = list(x)
        else:
            images = x

        detections = self.predict(images)
        return {}, detections

    @staticmethod
    def _coerce_to_chw_tensor(item: Any) -> torch.Tensor:
        """
        Convert common DLC detector inputs to a CHW uint8 torch tensor.

        Supported inputs:
          - str / Path
          - PIL.Image
          - np.ndarray
          - (image, context) tuples
        """
        if isinstance(item, tuple) and len(item) > 0:
            item = item[0]

        if isinstance(item, Image.Image):
            image = item.convert("RGB")

        elif isinstance(item, (str, Path)):
            with Image.open(item) as img:
                image = img.convert("RGB")

        elif isinstance(item, np.ndarray):
            arr = item

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            if arr.ndim != 3:
                raise ValueError(f"Expected image ndarray with shape HxWxC, got shape={arr.shape}.")

            if arr.shape[2] == 4:
                arr = arr[:, :, :3]

            if arr.shape[2] != 3:
                raise ValueError(f"Expected image ndarray with 3 or 4 channels, got shape={arr.shape}.")

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            image = Image.fromarray(arr, mode="RGB")

        else:
            raise TypeError(
                f"Unsupported image input type: {type(item)!r}. "
                "Supported: Path/str, PIL.Image, np.ndarray, or (image, context)."
            )

        arr = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _detection_to_context(det: DetectionResult) -> DetectorContext:
        """
        Convert one raw detector output into DLC detector context format.

        DetectorResult:
            boxes: XYXY absolute pixels
            scores: confidence scores

        DetectorContext:
            bboxes: XYWH absolute pixels
            bbox_scores: confidence scores
        """
        if "boxes" not in det:
            raise ValueError("Detection result must contain 'boxes'.")
        if "scores" not in det:
            raise ValueError("Detection result must contain 'scores'.")

        boxes = det["boxes"]
        scores = det["scores"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.asarray(boxes)

        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores)

        boxes_xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(boxes_xyxy) != len(scores):
            raise ValueError(f"Expected one score per box, got boxes={boxes_xyxy.shape}, scores={scores.shape}.")

        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)

        return {
            "bboxes": boxes_xywh.astype(np.float32, copy=False),
            "bbox_scores": scores.astype(np.float32, copy=False),
        }

    def inference(
        self,
        images,
        shelf_writer=None,
        *,
        show_progress: bool | None = None,
        progress_desc: str | None = None,
    ) -> list[DetectorContext]:
        """
        DetectorRunnerLike API.

        Converts external image inputs to tensors, calls predict(...), and converts
        raw XYXY detection results into DLC detector contexts with XYWH boxes.
        """
        _ = shelf_writer

        from tqdm.auto import tqdm

        image_list = list(images)
        config = getattr(self, "config", None)

        if show_progress is None:
            show_progress = bool(getattr(config, "show_progress", False))

        if progress_desc is None:
            progress_desc = getattr(self, "backend_name", self.__class__.__name__)

        batch_size = int(getattr(config, "batch_size", 1))
        batch_size = max(1, batch_size)

        outputs: list[DetectorContext] = []

        pbar = None
        if show_progress:
            pbar = tqdm(
                total=len(image_list),
                desc=progress_desc,
                unit="image",
                leave=True,
            )

        try:
            for start in range(0, len(image_list), batch_size):
                batch_items = image_list[start : start + batch_size]
                tensors = [self._coerce_to_chw_tensor(item) for item in batch_items]

                detections = self.predict(tensors)

                if len(detections) != len(batch_items):
                    raise ValueError(
                        f"Detector returned {len(detections)} outputs for "
                        f"{len(batch_items)} images in batch starting at index {start}."
                    )

                outputs.extend(self._detection_to_context(det) for det in detections)

                if pbar is not None:
                    pbar.update(len(batch_items))
        finally:
            if pbar is not None:
                pbar.close()

        if len(outputs) != len(image_list):
            raise ValueError(f"Detector returned {len(outputs)} outputs for {len(image_list)} images.")

        return outputs


class PrecomputedDetectorRunner:
    """
    Adapter that makes precomputed bbox entries behave like a detector runner.

    This is useful when you want to:
      - train a top-down pose model using precomputed detector outputs
      - run pose inference with saved bounding boxes
      - avoid running a live detector at all

    It implements the minimal `inference(images, shelf_writer=None)` method expected
    by the loader / dataset creation pathway.
    """

    def __init__(
        self,
        entries: list[BBoxEntry],
        *,
        target_format: BBoxFormat = "xywh",
        validate_image_paths: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.target_format = target_format
        self.validate_image_paths = validate_image_paths

        self._entries_by_path: dict[str, BBoxEntry] = {}
        for entry in self.entries:
            if entry.image_path is None:
                continue

            key = self._normalize_path_for_compare(entry.image_path)
            if key in self._entries_by_path:
                raise ValueError(f"Duplicate precomputed bbox entry for image_path={entry.image_path}")
            self._entries_by_path[key] = entry

    @staticmethod
    def _normalize_path_for_compare(path: Path | str) -> str:
        return Path(path).as_posix()

    @classmethod
    def from_bboxes(
        cls,
        bboxes: BBoxes,
        mode: EvalMode,
        *,
        target_format: BBoxFormat = "xywh",
        validate_image_paths: bool = False,
    ) -> PrecomputedDetectorRunner:
        return cls(
            entries=getattr(bboxes, mode),
            target_format=target_format,
            validate_image_paths=validate_image_paths,
        )

    @staticmethod
    def _normalize_path_for_compare(path: Path | str) -> str:
        return Path(path).as_posix().lower()

    @staticmethod
    def _extract_image_path(item) -> Path | None:
        if isinstance(item, tuple):
            image = item[0]
        else:
            image = item

        if isinstance(image, (str, Path)):
            return Path(image)

        return None

    def _find_entry_by_suffix(self, requested_path: Path) -> BBoxEntry | None:
        requested = self._normalize_path_for_compare(requested_path)

        matches = []
        for entry in self.entries:
            if entry.image_path is None:
                continue

            entry_path = self._normalize_path_for_compare(entry.image_path)

            if requested.endswith(entry_path) or entry_path.endswith(requested):
                matches.append(entry)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous precomputed bbox entries for requested image {requested_path}: "
                f"{[m.image_path for m in matches]}"
            )

        return None

    def inference(
        self,
        images,
        shelf_writer=None,
        *,
        show_progress: bool | None = None,
        progress_desc: str | None = None,
    ) -> list[DetectorContext]:
        """
        Return precomputed detector outputs aligned with the requested images.

        Args:
            images:
                Iterable of image inputs passed through DLC.
                Supported elements:
                - Path / str
                - (Path / str, context_dict)
                - np.ndarray / other non-path objects (order-only matching)

            shelf_writer:
                Accepted for compatibility, ignored.

            show_progress:
                Accepted for compatibility with BaseExternalDetector.inference(...).
                Precomputed lookup is usually fast, but if True, a progress bar is shown.

            progress_desc:
                Optional progress bar description.

        Returns:
            List of DLC detector contexts:
                [{"bboxes": ..., "bbox_scores": ...}, ...]
        """
        _ = shelf_writer

        from tqdm.auto import tqdm

        images = list(images)
        requested_paths = [self._extract_image_path(item) for item in images]

        outputs: list[DetectorContext] = []

        if progress_desc is None:
            progress_desc = self.__class__.__name__

        can_path_match = len(self._entries_by_path) > 0 and all(path is not None for path in requested_paths)

        if can_path_match:
            iterator = requested_paths
            if show_progress:
                iterator = tqdm(
                    requested_paths,
                    desc=progress_desc,
                    unit="image",
                    total=len(requested_paths),
                )

            for requested_path in iterator:
                assert requested_path is not None
                key = self._normalize_path_for_compare(requested_path)

                entry = self._entries_by_path.get(key)

                if entry is None:
                    # Optional useful fallback: match by filename/suffix when exact path differs.
                    entry = self._find_entry_by_suffix(requested_path)

                if entry is None:
                    raise ValueError(
                        f"No precomputed bbox entry found for requested image {requested_path}. "
                        f"Known entries include: {list(self._entries_by_path.keys())[:5]}"
                    )

                outputs.append(entry.to_detector_context(target_format=self.target_format))

            return outputs

        # Order-only fallback.
        # This is necessary for ndarray inputs or precomputed entries without paths.
        if self.validate_image_paths and any(path is not None for path in requested_paths):
            raise ValueError(
                "Cannot validate image paths because precomputed bbox entries do not contain "
                "image_path metadata for path-based lookup."
            )

        if len(images) > len(self.entries):
            raise ValueError(
                f"Got {len(images)} images but only {len(self.entries)} precomputed bbox entries "
                "are available for order-only matching."
            )

        entries = self.entries[: len(images)]
        iterator = entries
        if show_progress:
            iterator = tqdm(
                entries,
                desc=progress_desc,
                unit="image",
                total=len(entries),
            )

        for entry in iterator:
            outputs.append(entry.to_detector_context(target_format=self.target_format))

        return outputs


def _call_detector_inference(
    detector_runner: DetectorRunnerLike,
    images,
    *,
    shelf_writer=None,
    **kwargs: Any,
) -> list[DetectorContext]:
    """
    Call detector_runner.inference(...) while only passing optional kwargs
    supported by that runner.

    This keeps older DetectorRunnerLike implementations compatible.
    """
    inference_fn = detector_runner.inference
    signature = inspect.signature(inference_fn)

    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())

    if accepts_var_kwargs:
        supported_kwargs = kwargs
    else:
        supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    return inference_fn(
        images,
        shelf_writer=shelf_writer,
        **supported_kwargs,
    )


def precompute_detector_bboxes(
    loader: Loader,
    detector_runner: DetectorRunnerLike,
    output_file: str | Path,
    modes: tuple[str, ...] = ("train", "test"),
    *,
    bbox_format: str = "xywh",
    show_progress: bool | None = None,
) -> BBoxes:
    """
    Run a detector runner on all images for the requested modes and save the results
    to a BBoxes JSON artifact.

    Args:
        show_progress:
            If None, let the detector decide.
            If True/False, override detector progress behavior when supported.
    """
    output_file = Path(output_file)

    result = {}
    for mode in modes:
        if hasattr(loader, "get_image_paths"):
            image_paths = [Path(p) for p in loader.get_image_paths(mode)]  # type: ignore[attr-defined]
        else:
            image_paths = [Path(p) for p in loader.image_filenames(mode)]

        outputs = _call_detector_inference(
            detector_runner,
            image_paths,
            show_progress=show_progress,
            progress_desc=f"Computing dataset detections for: [{mode}]",
        )

        if len(outputs) != len(image_paths):
            raise ValueError(f"Detector returned {len(outputs)} outputs for {len(image_paths)} {mode} images.")

        result[mode] = [
            BBoxEntry.from_detector_context(
                out,
                image_path=img_path,
                bbox_format=bbox_format,
            )
            for img_path, out in zip(image_paths, outputs, strict=False)
        ]

    bboxes = BBoxes(**result)
    bboxes.dump_json(output_file)
    return bboxes


def build_precomputed_detector_runner_from_config(
    model_cfg: dict,
    mode: str,
    *,
    target_format: BBoxFormat = "xywh",
    validate_image_paths: bool = False,
) -> PrecomputedDetectorRunner | None:
    """
    Build a precomputed detector runner from model_cfg["data"]["precomputed_bboxes"].
    Returns None if no precomputed bbox file is configured.
    """
    data_cfg = model_cfg.get("data", {})
    bbox_file = data_cfg.get("precomputed_bboxes")
    if bbox_file is None:
        return None

    bboxes = BBoxes.from_file(Path(bbox_file))
    return PrecomputedDetectorRunner.from_bboxes(
        bboxes,
        mode=mode,
        target_format=target_format,
        validate_image_paths=validate_image_paths,
    )
