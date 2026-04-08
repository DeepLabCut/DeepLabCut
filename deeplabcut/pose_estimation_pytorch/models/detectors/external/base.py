from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypedDict

import torch
import torch.nn as nn

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
    inference-oriented and usually do not participate in DLC training,
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
