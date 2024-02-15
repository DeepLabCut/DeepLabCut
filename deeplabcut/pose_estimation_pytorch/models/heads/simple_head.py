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
        self,
        channels: list[int],
        kernel_size: list[int],
        strides: list[int],
        final_conv: dict | None = None,
    ) -> None:
        """
        Args:
            channels: List containing the number of input and output channels for each
                deconvolutional layer.
            kernel_size: List containing the kernel size for each deconvolutional layer.
            strides: List containing the stride for each deconvolutional layer.
            final_conv: Configuration for a conv layer after the deconvolutional layers,
                if one should be added. Must have keys "out_channels" and "kernel_size".
        """
        super().__init__()
        if not (len(channels) == len(kernel_size) + 1 == len(strides) + 1):
            raise ValueError(
                "Incorrect DeconvModule configuration: there should be one more number"
                f" of channels than kernel_sizes and strides, found {len(channels)} "
                f"channels, {len(kernel_size)} kernels and {len(strides)} strides."
            )

        in_channels = channels[0]
        self.deconv_layers = nn.Identity()
        if len(kernel_size) > 0:
            self.deconv_layers = nn.Sequential(
                *self._make_layers(in_channels, channels[1:], kernel_size, strides)
            )

        self.final_conv = nn.Identity()
        if final_conv:
            self.final_conv = nn.Conv2d(
                in_channels=channels[-1],
                out_channels=final_conv["out_channels"],
                kernel_size=final_conv["kernel_size"],
                stride=1,
            )

    @staticmethod
    def _make_layers(
        in_channels: int,
        out_channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
    ) -> list[nn.Module]:
        """
        Helper function to create the deconvolutional layers.

        Args:
            in_channels: number of input channels to the module
            out_channels: number of output channels of each layer
            kernel_sizes: size of the deconvolutional kernel
            strides: stride for the convolution operation

        Returns:
            the deconvolutional layers
        """
        layers = []
        for out_channels, k, s in zip(out_channels, kernel_sizes, strides):
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        return layers[:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HeatmapHead

        Args:
            x: input tensor

        Returns:
            out: output tensor
        """
        x = self.deconv_layers(x)
        x = self.final_conv(x)
        return x
