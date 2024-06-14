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
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

BACKBONES = Registry("backbones", build_func=build_from_cfg)


class BaseBackbone(ABC, nn.Module):
    """Base Backbone class for pose estimation.

    Attributes:
        stride: the stride for the backbone
        freeze_bn_weights: freeze weights of batch norm layers during training
        freeze_bn_stats: freeze stats of batch norm layers during training
    """

    def __init__(
        self,
        stride: int | float,
        freeze_bn_weights: bool = True,
        freeze_bn_stats: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.freeze_bn_weights = freeze_bn_weights
        self.freeze_bn_stats = freeze_bn_stats

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract method for the forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            a feature map for the input, of shape (batch_size, c', h', w')
        """
        pass

    def freeze_batch_norm_layers(self) -> None:
        """Freezes batch norm layers

        Running mean + var are always given to F.batch_norm, except when the layer is
        in `train` mode and track_running_stats is False, see
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
        So to 'freeze' the running stats, the only way is to set the layer to "eval"
        mode.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if self.freeze_bn_weights:
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                if self.freeze_bn_stats:
                    module.eval()

    def train(self, mode: bool = True) -> None:
        """Sets the module in training or evaluation mode.

        Args:
            mode: whether to set training mode (True) or evaluation mode (False)
        """
        super().train(mode)
        if self.freeze_bn_weights or self.freeze_bn_stats:
            self.freeze_batch_norm_layers()
