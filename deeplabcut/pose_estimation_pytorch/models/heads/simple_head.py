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
from deeplabcut.pose_estimation_pytorch.models.heads.base import (
    BaseHead,
    HEADS,
    WeightConversionMixin,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.models.weight_init import BaseWeightInitializer


@HEADS.register_module
class HeatmapHead(WeightConversionMixin, BaseHead):
    """Deconvolutional head to predict maps from the extracted features.

    This class implements a simple deconvolutional head to predict maps from the
    extracted features.

    Args:
        predictor: The predictor used to transform heatmaps into keypoints.
        target_generator: The module to generate target heatmaps from keypoints.
        criterion: The loss criterion(s) for the head.
        aggregator: The loss aggregator to use, if multiple criterions are used.
        heatmap_config: The configuration for the heatmap outputs of the head.
        locref_config: The configuration for the location refinement outputs (None if
            no location refinement should be used).
        weight_init: The way to initialize weights for the head. If None, default
            PyTorch initialization is used. Otherwise, a BaseWeightInitializer can be
            given (or a configuration for a BaseWeightInitializer). To initialize
            the weights with a normal distribution, you could pass
            ``weight_init="normal"`` (which initializes weights using a Normal
            distribution 0.001 and biases with 0), or you could pass ``weight_init={
            type="normal", std=0.01}`` to change the standard deviation used. All
            BaseWeightInitializers are defined in deeplabcut/pose_estimation_pytorch/
            models/weight_init.py.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion] | BaseCriterion,
        aggregator: BaseLossAggregator | None,
        heatmap_config: dict,
        locref_config: dict | None = None,
        weight_init: str | dict | BaseWeightInitializer | None = None,
    ) -> None:
        heatmap_head = DeconvModule(**heatmap_config)
        locref_head = None
        if locref_config is not None:
            locref_head = DeconvModule(**locref_config)

            # check that the heatmap and locref modules have the same stride
            if heatmap_head.stride != locref_head.stride:
                raise ValueError(
                    f"Invalid model config: Your heatmap and locref need to have the "
                    f"same stride (found {heatmap_head.stride}, "
                    f"{locref_head.stride}). Please check your config (found "
                    f"heatmap_config={heatmap_config}, locref_config={locref_config}"
                )

        super().__init__(
            heatmap_head.stride,
            predictor,
            target_generator,
            criterion,
            aggregator,
            weight_init,
        )
        self.heatmap_head = heatmap_head
        self.locref_head = locref_head
        self._init_weights()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {"heatmap": self.heatmap_head(x)}
        if self.locref_head is not None:
            outputs["locref"] = self.locref_head(x)
        return outputs

    @staticmethod
    def convert_weights(
        state_dict: dict[str, torch.Tensor],
        module_prefix: str,
        conversion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Converts pre-trained weights to be fine-tuned on another dataset

        Args:
            state_dict: the state dict for the pre-trained model
            module_prefix: the prefix for weights in this head (e.g., 'heads.bodypart.')
            conversion: the mapping of old indices to new indices
        """
        state_dict = DeconvModule.convert_weights(
            state_dict,
            f"{module_prefix}heatmap_head.",
            conversion,
        )

        locref_conversion = torch.stack(
            [2 * conversion, 2 * conversion + 1],
            dim=1,
        ).reshape(-1)
        state_dict = DeconvModule.convert_weights(
            state_dict,
            f"{module_prefix}locref_head.",
            locref_conversion,
        )
        return state_dict


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
        head_stride = 1
        self.deconv_layers = nn.Identity()
        if len(kernel_size) > 0:
            self.deconv_layers = nn.Sequential(
                *self._make_layers(in_channels, channels[1:], kernel_size, strides)
            )
            for s in strides:
                head_stride *= s

        self.stride = head_stride
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

    @staticmethod
    def convert_weights(
        state_dict: dict[str, torch.Tensor],
        module_prefix: str,
        conversion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Converts pre-trained weights to be fine-tuned on another dataset

        Args:
            state_dict: the state dict for the pre-trained model
            module_prefix: the prefix for weights in this head (e.g., 'heads.bodypart')
            conversion: the mapping of old indices to new indices
        """
        if f"{module_prefix}final_conv.weight" in state_dict:
            # has final convolution
            weight_key = f"{module_prefix}final_conv.weight"
            bias_key = f"{module_prefix}final_conv.bias"
            state_dict[weight_key] = state_dict[weight_key][conversion]
            state_dict[bias_key] = state_dict[bias_key][conversion]
            return state_dict

        # get the last deconv layer of the net
        next_index = 0
        while f"{module_prefix}deconv_layers.{next_index}.weight" in state_dict:
            next_index += 1
        last_index = next_index - 1

        # if there are deconv layers for this module prefix (there might not be,
        # e.g., when there are no location refinement layers in a heatmap head)
        if last_index >= 0:
            weight_key = f"{module_prefix}deconv_layers.{last_index}.weight"
            bias_key = f"{module_prefix}deconv_layers.{last_index}.bias"

            # for ConvTranspose2d, the weight shape is (in_channels, out_channels, ...)
            # while it's (out_channels, in_channels, ...) for Conv2d
            state_dict[weight_key] = state_dict[weight_key][:, conversion]
            state_dict[bias_key] = state_dict[bias_key][conversion]

        return state_dict
