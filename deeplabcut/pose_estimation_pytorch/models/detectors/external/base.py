from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, TypedDict

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.data.base import DetectorRunnerLike, Loader
from deeplabcut.pose_estimation_pytorch.data.bboxes import BBoxEntry, BBoxes, BBoxFormat, DetectorContext, EvalMode
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

    These detectors expose a canonical inference API:
        predict(images) -> list[DetectionResult]

    and a forward() shim for compatibility with DLC inference runners:
        forward(images, targets=None) -> ({}, detections)
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
            # Assume batched BCHW tensor -> convert to list[CHW]
            images = list(x)
        else:
            images = x

        detections = self.predict(images)
        return {}, detections


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

    def inference(self, images, shelf_writer=None) -> list[DetectorContext]:
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

        Returns:
            List of DLC detector contexts:
                [{"bboxes": ..., "bbox_scores": ...}, ...]
        """
        requested_paths: list[Path | None] = []

        for item in images:
            if isinstance(item, tuple):
                image = item[0]
            else:
                image = item

            if isinstance(image, (str, Path)):
                requested_paths.append(Path(image))
            else:
                # For array inputs, we cannot path-match — use order only
                requested_paths.append(None)

        if len(requested_paths) != len(self.entries):
            raise ValueError(f"Got {len(requested_paths)} images but {len(self.entries)} precomputed bbox entries.")

        outputs: list[DetectorContext] = []

        for requested_path, entry in zip(requested_paths, self.entries, strict=False):
            if self.validate_image_paths and requested_path is not None and entry.image_path is not None:
                if self._normalize_path_for_compare(entry.image_path) != self._normalize_path_for_compare(
                    requested_path
                ):
                    raise ValueError(
                        f"Precomputed bbox entry path mismatch: expected {requested_path}, got {entry.image_path}"
                    )

            outputs.append(entry.to_detector_context(target_format=self.target_format))

        return outputs


def precompute_detector_bboxes(
    loader: Loader,
    detector_runner: DetectorRunnerLike,
    output_file: str | Path,
    modes: tuple[str, ...] = ("train", "test"),
    *,
    bbox_format: str = "xywh",
) -> BBoxes:
    """
    Run a detector runner on all images for the requested modes and save the results
    to a BBoxes JSON artifact.

    The saved artifact is intended to be reused later for training a top-down pose
    model without rerunning the detector.
    """
    output_file = Path(output_file)

    result = {}
    for mode in modes:
        image_paths = [Path(p) for p in loader.get_image_paths(mode)]
        outputs = detector_runner.inference(image_paths)

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
    target_format: str = "xywh",
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
