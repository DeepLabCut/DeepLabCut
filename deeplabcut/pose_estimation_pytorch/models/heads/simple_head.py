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

import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.heads.base import HEADS
from einops import rearrange
from timm.layers import trunc_normal_

from .base import BaseHead


@HEADS.register_module
class SimpleHead(BaseHead):
    """
    Deconvolutional head to predict maps from the extracted features.
    This class implements a simple deconvolutional head to predict maps from the extracted features.
    """

    def __init__(
        self, channels: list, kernel_size: list, strides: list, pretrained: str = None
    ) -> None:
        """Summary
        Constructor of the SimpleHead object.
        Loads the data.

        Args:
            channels: list containing the number of input and output channels for each deconvolutional layer.
            kernel_size: list containing the kernel size for each deconvolutional layer.
            strides: list containing the stride for each deconvolutional layer.
            pretrained: path to a pretrained model checkpoint. Defaults to None. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides

        if len(kernel_size) == 1:
            self.model = self._make_layer(
                channels[0], channels[1], kernel_size[0], strides[0]
            )
        else:
            layers = []
            for i in range(len(channels) - 1):
                up_layer = self._make_layer(
                    channels[i], channels[i + 1], kernel_size[i], strides[i]
                )
                layers.append(up_layer)
                if i < len(channels) - 2:
                    layers.append(nn.ReLU())
            self.model = nn.Sequential(*layers)

        self._init_weights(pretrained)

    def _make_layer(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> torch.nn.ConvTranspose2d:
        """Summary:
        Helper function to create a deconvolutional layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of the deconvolutional kernel
            stride: stride for the covolution operation

        Returns:
            upsample_layer: the deconvolutional layer.
        """
        upsample_layer = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        return upsample_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Summary:
        Forward pass of the SimpleHead object.

        Args:
            x: input tensor

        Returns:
            out: output tensor
        """
        out = self.model(x)

        return out


@HEADS.register_module
class TransformerHead(BaseHead):
    """
    Transformer Head module to predict heatmaps using a transformer-based approach
    """

    def __init__(
        self,
        dim: int,
        hidden_heatmap_dim: int,
        heatmap_dim: int,
        apply_multi: bool,
        heatmap_size: tuple,
        apply_init: bool,
    ):
        """Summary:
            Given the output of a transformer neck, this head applies an mlp head to compute the heatmaps
        Args:
            dim: Dimension of the input features.
            hidden_heatmap_dim: Dimension of the hidden features in the MLP head.
            heatmap_dim: Dimension of the output heatmaps.
            apply_multi: If True, apply a multi-layer perceptron (MLP) with LayerNorm
                                to generate heatmaps. If False, directly apply a single linear
                                layer for heatmap prediction.
            heatmap_size: Tuple (height, width) representing the size of the output
                                  heatmaps.
            apply_init: If True, apply weight initialization to the module's layers.

        Returns:
            None
        """
        super().__init__()
        self.mlp_head = (
            nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, hidden_heatmap_dim),
                nn.LayerNorm(hidden_heatmap_dim),
                nn.Linear(hidden_heatmap_dim, heatmap_dim),
            )
            if (dim * 3 <= hidden_heatmap_dim * 0.5 and apply_multi)
            else nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, heatmap_dim))
        )
        self.heatmap_size = heatmap_size
        # trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Summary
        Custom weight initialization for linear and layer normalization layers.

        Args:
            m: module to initialize

        Returns:
            None
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Summary:
        Forward pass of the TransformerHead class

        Args:
            x: input tensor

        Returns:
            x: output tensor containing predicted heatmaps
        """
        x = self.mlp_head(x)
        x = rearrange(
            x,
            "b c (p1 p2) -> b c p1 p2",
            p1=self.heatmap_size[0],
            p2=self.heatmap_size[1],
        )

        return x
