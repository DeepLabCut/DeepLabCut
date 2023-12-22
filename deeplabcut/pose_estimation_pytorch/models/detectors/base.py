#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

DETECTORS = Registry("detectors", build_func=build_from_cfg)


class BaseDetector(ABC, nn.Module):
    """
    Definition of the class BaseDetector object.
    This is an abstract class defining the common structure and inference for detectors.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """
        Forward pass of the detector

        Args:
            x: images to be processed
            targets: ground-truth boxes present in each images

        Returns:
            losses: {'loss_name': loss_value}
            detections: for each of the b images, {"boxes": bounding_boxes}
        """
        pass

    @abstractmethod
    def get_target(self, labels: dict) -> list[dict]:
        """
        Get the target for training the detector

        Args:
            labels: annotations containing keypoints, bounding boxes, etc.

        Returns:
            list of dictionaries, each representing target information for a single annotation.
        """
        pass
