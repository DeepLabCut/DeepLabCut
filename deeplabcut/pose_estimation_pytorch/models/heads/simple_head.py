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

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.heads.base import BaseHead, HEADS
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator


@HEADS.register_module
class HeatmapHead(BaseHead):
    """
    Deconvolutional head to predict maps from the extracted features.
    This class implements a simple deconvolutional head to predict maps from the extracted features.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion] | BaseCriterion,
        aggregator: BaseLossAggregator | None,
        heatmap_config: dict,
        locref_config: dict | None = None,
    ) -> None:
        super().__init__(predictor, target_generator, criterion, aggregator)
        self.heatmap_head = DeconvModule(**heatmap_config)
        self.locref_head = None
        if locref_config is not None:
            self.locref_head = DeconvModule(**locref_config)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {"heatmap": self.heatmap_head(x)}
        if self.locref_head is not None:
            outputs["locref"] = self.locref_head(x)
        return outputs


class DeconvModule(nn.Module):
    """
    Deconvolutional module to predict maps from the extracted features.
    """

    def __init__(
        self, channels: list[int], kernel_size: list[int], strides: list[int]
    ) -> None:
        """
        Args:
            channels: list containing the number of input and output channels for each deconvolutional layer.
            kernel_size: list containing the kernel size for each deconvolutional layer.
            strides: list containing the stride for each deconvolutional layer.
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

    def _make_layer(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> torch.nn.ConvTranspose2d:
        """
        Helper function to create a deconvolutional layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of the deconvolutional kernel
            stride: stride for the convolution operation

        Returns:
            upsample_layer: the deconvolutional layer.
        """
        upsample_layer = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        return upsample_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleHead object.

        Args:
            x: input tensor

        Returns:
            out: output tensor
        """
        return self.model(x)
