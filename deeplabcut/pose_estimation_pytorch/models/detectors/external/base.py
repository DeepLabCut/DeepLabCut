from __future__ import annotations

import inspect
import logging
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
    EmptyBBoxPolicy,
    EvalMode,
    RecomputePolicy,
    _xyxy_to_xywh,
    index_bbox_entries_by_path,
    normalize_bbox_image_path,
    should_recompute_bbox_entry,
    validate_bbox_entry,
)
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

logger = logging.getLogger(__name__)


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


def validate_precomputed_bboxes_for_loader(
    loader: Loader,
    bbox_file: str | Path,
    *,
    required_modes: tuple[str, ...] = ("train", "test"),
    target_format: BBoxFormat = "xywh",
    require_image_paths: bool = True,
    allow_empty_bboxes: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Validate that a precomputed bbox artifact is complete and compatible with a loader.

    This is intended as a preflight check before training a top-down pose model with
    external/precomputed detector boxes.

    Important:
        Entries with zero bboxes can be valid, e.g. true empty/no-animal frames.
        Missing entries are not valid for training because they mean the artifact was
        not computed for all images in the requested split.

    Args:
        loader:
            DLC loader for the project/shuffle.
        bbox_file:
            Path to precomputed_bboxes.json.
        required_modes:
            Modes that must be complete. For normal training, use ("train", "test").
        target_format:
            Format to validate after conversion. Usually "xywh".
        require_image_paths:
            If True, every BBoxEntry must contain image_path metadata.
        allow_empty_bboxes:
            If False, entries with zero boxes are errors. Usually keep True.

    Returns:
        Per-mode summary dictionary.

    Raises:
        FileNotFoundError:
            If the bbox artifact does not exist.
        ValueError:
            If the artifact is incomplete or malformed.
    """
    bbox_file = Path(bbox_file)

    if not bbox_file.exists():
        raise FileNotFoundError(
            f"Precomputed bbox artifact not found: {bbox_file}. Run precompute_detector_bboxes(...) before training."
        )

    bboxes = BBoxes.from_file(bbox_file)

    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, dict[str, int]] = {}

    def normalize_path(path: str | Path) -> str:
        return Path(path).as_posix().lower()

    def loader_image_paths(mode: str) -> list[Path]:
        if hasattr(loader, "get_image_paths"):
            return [Path(p) for p in loader.get_image_paths(mode)]  # type: ignore[attr-defined]
        return [Path(p) for p in loader.image_filenames(mode)]

    for mode in required_modes:
        image_paths = loader_image_paths(mode)
        entries = list(getattr(bboxes, mode, []))

        mode_summary = {
            "expected_images": len(image_paths),
            "bbox_entries": len(entries),
            "entries_with_image_path": sum(e.image_path is not None for e in entries),
            "entries_without_bboxes": 0,
            "total_bboxes": 0,
        }

        if len(entries) != len(image_paths):
            errors.append(
                f"[{mode}] Expected {len(image_paths)} bbox entries, found {len(entries)}. "
                "This usually means the artifact was computed for the wrong split, "
                "only partially computed, or computed with modes missing. "
                f"Re-run precompute_detector_bboxes(..., modes={required_modes!r})."
            )

        entries_by_path: dict[str, BBoxEntry] = {}
        duplicate_paths: list[str] = []

        for entry in entries:
            if entry.image_path is None:
                continue

            key = normalize_path(entry.image_path)
            if key in entries_by_path:
                duplicate_paths.append(key)
            entries_by_path[key] = entry

        if duplicate_paths:
            errors.append(
                f"[{mode}] Duplicate bbox entries found for image_path values. Examples: {duplicate_paths[:5]}"
            )

        if require_image_paths and entries:
            missing_metadata = [i for i, e in enumerate(entries) if e.image_path is None]
            if missing_metadata:
                errors.append(
                    f"[{mode}] {len(missing_metadata)} bbox entries are missing image_path metadata. "
                    "Path metadata is required to verify that the bbox artifact matches this project/shuffle."
                )

        if entries_by_path:
            expected_keys = {normalize_path(p) for p in image_paths}
            actual_keys = set(entries_by_path)

            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)

            if missing:
                errors.append(
                    f"[{mode}] Missing bbox entries for {len(missing)} loader images. Examples: {missing[:5]}"
                )

            if extra:
                warnings.append(
                    f"[{mode}] Artifact contains {len(extra)} bbox entries not used by this loader/shuffle. "
                    f"Examples: {extra[:5]}"
                )

        for i, entry in enumerate(entries):
            try:
                context = entry.to_detector_context(target_format=target_format)
            except Exception as exc:
                errors.append(f"[{mode}] Entry {i} failed conversion to {target_format!r}: {exc}")
                continue

            boxes = np.asarray(
                context.get("bboxes", np.zeros((0, 4), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 4)

            scores = np.asarray(
                context.get("bbox_scores", np.ones((len(boxes),), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)

            mode_summary["total_bboxes"] += len(boxes)

            if len(boxes) == 0:
                mode_summary["entries_without_bboxes"] += 1
                if not allow_empty_bboxes:
                    errors.append(f"[{mode}] Entry {i} has no bboxes.")
                continue

            if len(scores) != len(boxes):
                errors.append(f"[{mode}] Entry {i} has {len(boxes)} boxes but {len(scores)} bbox_scores.")

            if not np.isfinite(boxes).all():
                errors.append(f"[{mode}] Entry {i} contains non-finite bbox coordinates.")

            if not np.isfinite(scores).all():
                errors.append(f"[{mode}] Entry {i} contains non-finite bbox scores.")

            if target_format == "xywh":
                invalid_dims = (boxes[:, 2] < 0) | (boxes[:, 3] < 0)
                if np.any(invalid_dims):
                    errors.append(
                        f"[{mode}] Entry {i} contains {int(invalid_dims.sum())} boxes with negative width/height."
                    )

        summary[mode] = mode_summary

    for warning in warnings:
        logging.warning(warning)

    if errors:
        message = [
            "Invalid or incomplete precomputed bbox artifact for external top-down training.",
            "",
            f"Artifact: {bbox_file}",
            f"Required modes: {required_modes}",
            "",
            "Problems found:",
            *[f"  - {error}" for error in errors],
            "",
            "How to fix:",
            f"  Run precompute_detector_bboxes(..., output_file={str(bbox_file)!r}, modes={required_modes!r}).",
            "",
            "Note:",
            "  Entries with zero bboxes are allowed by default because they can represent true empty/no-animal frames.",
            "  Missing entries are not allowed because the data loader cannot align detector boxes to all images.",
        ]
        raise ValueError("\n".join(message))

    return summary


def precompute_detector_bboxes(
    loader: Loader,
    detector_runner: DetectorRunnerLike,
    output_file: str | Path,
    modes: tuple[str, ...] = ("train", "test"),
    *,
    bbox_format: BBoxFormat = "xywh",
    show_progress: bool | None = None,
    recompute: RecomputePolicy = "resume",
    empty_policy: EmptyBBoxPolicy = "valid",
    validate_image_paths: bool = True,
    min_existing_score: float | None = None,
    detector_metadata: dict[str, Any] | None = None,
    strict: bool = True,
) -> BBoxes:
    """
    Run a detector runner on requested images and save a reusable BBoxes JSON artifact.

    Existing entries can be reused or selectively recomputed.

    Args:
        recompute:
            - "missing": recompute only missing entries
            - "invalid": recompute only invalid existing entries
            - "missing_or_invalid": recompute missing or invalid entries
            - "all": recompute everything
            - "none": never run detector; validate/reuse only
            - "resume": user-friendly default; reuse valid entries, recompute missing/invalid entries

        empty_policy:
            - "valid": zero-box entries are valid
            - "invalid": zero-box entries are invalid
            - "recompute": zero-box entries are structurally valid, but selected for
              recomputation when recompute includes invalid entries

        validate_image_paths:
            If True, reuse existing entries by image_path. Existing entries without
            image_path metadata are treated as missing.

            If False, fall back to order-based reuse when path metadata is absent.

        min_existing_score:
            If set, existing entries with boxes but max score below this value are
            considered invalid/recompute candidates.

        detector_metadata:
            Optional metadata to store in the artifact under metadata["detector"].

        strict:
            If True, raise when an entry is missing/invalid and recompute policy does
            not allow recomputing it.
    """
    output_file = Path(output_file)
    existing_bboxes = BBoxes.from_file(output_file, missing_ok=True)

    if detector_metadata is not None:
        existing_bboxes.metadata["detector"] = detector_metadata

    result: dict[str, list[BBoxEntry]] = {
        "train": list(existing_bboxes.train),
        "test": list(existing_bboxes.test),
    }

    for mode in modes:
        if hasattr(loader, "get_image_paths"):
            image_paths = [Path(p) for p in loader.get_image_paths(mode)]  # type: ignore[attr-defined]
        else:
            image_paths = [Path(p) for p in loader.image_filenames(mode)]

        existing_entries = list(getattr(existing_bboxes, mode, []))
        entries_by_path, duplicate_paths = index_bbox_entries_by_path(existing_entries)

        if duplicate_paths:
            message = (
                f"[{mode}] Found duplicate image_path entries in existing bbox artifact. "
                f"Examples: {duplicate_paths[:5]}"
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

        final_entries: list[BBoxEntry | None] = [None] * len(image_paths)
        recompute_paths: list[Path] = []
        recompute_indices: list[int] = []
        errors: list[str] = []

        for i, image_path in enumerate(image_paths):
            key = normalize_bbox_image_path(image_path)

            entry: BBoxEntry | None = None

            if validate_image_paths:
                # Strict path-based reuse. If old entries have no image_path metadata,
                # they are treated as missing and can be recomputed.
                entry = entries_by_path.get(key)
            else:
                # Prefer path lookup when available, otherwise allow order-based reuse.
                entry = entries_by_path.get(key)
                if entry is None and i < len(existing_entries):
                    entry = existing_entries[i]

            exists = entry is not None
            valid = False
            reason = "missing bbox entry"
            empty = False

            if exists:
                valid, reason, empty = validate_bbox_entry(
                    entry,
                    target_format=bbox_format,
                    empty_policy=empty_policy,
                    min_existing_score=min_existing_score,
                )

            recompute_this = should_recompute_bbox_entry(
                exists=exists,
                valid=valid,
                empty=empty,
                recompute=recompute,
                empty_policy=empty_policy,
            )

            if recompute_this:
                recompute_paths.append(image_path)
                recompute_indices.append(i)
                continue

            if not exists:
                errors.append(
                    f"[{mode}] Missing bbox entry for image {image_path}. "
                    f"Set recompute='missing' or recompute='missing_or_invalid' to compute it."
                )
                continue

            if not valid:
                errors.append(
                    f"[{mode}] Invalid bbox entry for image {image_path}: {reason}. "
                    f"Set recompute='invalid' or recompute='missing_or_invalid' to recompute it."
                )
                continue

            final_entries[i] = entry

        if errors and strict:
            message = [
                "Cannot build complete precomputed bbox artifact.",
                f"Artifact: {output_file}",
                f"Mode: {mode}",
                f"Recompute policy: {recompute}",
                "",
                "Problems found:",
                *[f"  - {error}" for error in errors[:20]],
            ]

            if len(errors) > 20:
                message.append(f"  ... and {len(errors) - 20} more problems")

            message.extend(
                [
                    "",
                    "How to fix:",
                    "  Re-run precompute_detector_bboxes(...) with "
                    "recompute='missing_or_invalid', or recompute='all' if you want a fresh artifact.",
                ]
            )

            raise ValueError("\n".join(message))

        if len(recompute_paths) > 0:
            if recompute == "none":
                raise ValueError(f"[{mode}] {len(recompute_paths)} entries need recomputation, but recompute='none'.")

            outputs = _call_detector_inference(
                detector_runner,
                recompute_paths,
                show_progress=show_progress,
                progress_desc=f"Computing dataset detections for: [{mode}]",
            )

            if len(outputs) != len(recompute_paths):
                raise ValueError(
                    f"Detector returned {len(outputs)} outputs for {len(recompute_paths)} {mode} images to recompute."
                )

            for i, image_path, out in zip(recompute_indices, recompute_paths, outputs, strict=False):
                final_entries[i] = BBoxEntry.from_detector_context(
                    out,
                    image_path=image_path,
                    bbox_format=bbox_format,
                )

        missing_after_merge = [image_paths[i] for i, entry in enumerate(final_entries) if entry is None]
        if missing_after_merge:
            raise ValueError(
                f"[{mode}] Could not build a complete bbox artifact. "
                f"{len(missing_after_merge)} entries are still missing. "
                f"Examples: {missing_after_merge[:5]}"
            )

        result[mode] = [entry for entry in final_entries if entry is not None]

        reused = len(image_paths) - len(recompute_paths)
        logger.info(
            "Precomputed bboxes [%s]: reused %d entries, recomputed %d entries",
            mode,
            reused,
            len(recompute_paths),
        )

    bboxes = BBoxes(
        schema_version=existing_bboxes.schema_version,
        metadata=existing_bboxes.metadata,
        train=result["train"],
        test=result["test"],
    )
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
