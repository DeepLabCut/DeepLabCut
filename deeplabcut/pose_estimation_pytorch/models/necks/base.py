#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from abc import ABC, abstractmethod

import torch

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

NECKS = Registry("necks", build_func=build_from_cfg)


class BaseNeck(ABC, torch.nn.Module):
    """Base Neck class for pose estimation"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Abstract method for the forward pass through the Neck.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        pass

    def _init_weights(self, pretrained: str):
        """Initialize the Neck with pretrained weights.

        Args:
            pretrained: Path to the pretrained weights.

        Returns:
            None
        """
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained))
