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
    def forward(self, x: torch.Tensor) -> None:
        """Summary:
        Forward pass of the detector

        Args:
            x: input tensor representing the image

        Returns:
            See base class.
        """
        pass

    @abstractmethod
    def get_target(self, annotations) -> None:
        """Summary:
        Get the target for training the detector

        Args:
            annotations: annotations containing keypoints, bounding boxes, etc.

        Returns:
            None
        """
        pass

    def _init_weights(self, pretrained: bool) -> None:
        """Summary:
        Initialize weights for the detector

        Args:
            pretrained: whether to use pretrained weights.

        Returns:
            None
        """
        if not pretrained:
            pass
        else:
            self.model.load_state_dict(torch.load(pretrained))
