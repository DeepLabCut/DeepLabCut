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
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

BACKBONES = Registry("backbones", build_func=build_from_cfg)


class BaseBackbone(ABC, torch.nn.Module):
    """Base Backbone class for pose estimation.

    Attributes:
        batch_norm_on: Indicates whether batch normalization is activated during training.
    """

    def __init__(self):
        """Initialize the BaseBackbone."""
        super().__init__()
        self.batch_norm_on = False

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Abstract method for the forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor.
        """
        pass

    def _init_weights(self, pretrained: str = None):
        """Initialize the backbone with pretrained weights.

        Args:
            pretrained: Path to the pretrained weights.
                                        Defaults to None.

        Returns:
            None
        """
        if not pretrained:
            pass
        elif pretrained.startswith("http") or pretrained.startswith("ftp"):
            state_dict = torch.hub.load_state_dict_from_url(pretrained)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(torch.load(pretrained), strict=False)

    def activate_batch_norm(self, activation: bool = False):
        """Activate or deactivate batch normalization layers during training.

        Args:
            activation: Activate or deactivate batch normalization.
                                         Defaults to False.

        Returns:
            None
        """
        self.batch_norm_on = activation

    def train(self, mode=True):
        """Set the training mode with optional batch normalization activation.

        Args:
            mode: Training mode. Defaults to True.

        Returns:
            None

        Note:
            Batch Norm should not be on for small batch sizes.
        """
        super().train(mode)

        if not self.batch_norm_on:
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        return
