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
from deeplabcut.pose_estimation_pytorch.models.modules.conv_block import (
    AdaptBlock,
    BaseBlock,
    BasicBlock,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.models.weight_init import BaseWeightInitializer


@HEADS.register_module
class DEKRHead(BaseHead):
    """
    DEKR head based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang, CVPR 2021
    Code based on:
        https://github.com/HRNet/DEKR
    """

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion],
        aggregator: BaseLossAggregator,
        heatmap_config: dict,
        offset_config: dict,
        weight_init: str | dict | BaseWeightInitializer | None = "dekr",
        stride: int | float = 1,  # head stride - should always be 1 for DEKR
    ) -> None:
        super().__init__(
            stride, predictor, target_generator, criterion, aggregator, weight_init
        )
        self.heatmap_head = DEKRHeatmap(**heatmap_config)
        self.offset_head = DEKROffset(**offset_config)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"heatmap": self.heatmap_head(x), "offset": self.offset_head(x)}


class DEKRHeatmap(nn.Module):
    """
    DEKR head to compute the heatmaps corresponding to keypoints based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang, CVPR 2021
    Code based on:
        https://github.com/HRNet/DEKR
    """

    def __init__(
        self,
        channels: tuple[int],
        num_blocks: int,
        dilation_rate: int,
        final_conv_kernel: int,
        block: type(BaseBlock) = BasicBlock,
    ) -> None:
        """Summary:
        Constructor of the HeatmapDEKRHead.
        Loads the data.

        Args:
            channels: tuple containing the number of channels for the head.
            num_blocks: number of blocks in the head
            dilation_rate: dilation rate for the head
            final_conv_kernel: kernel size for the final convolution
            block: type of block to use in the head. Defaults to BasicBlock.

        Returns:
            None

        Examples:
            channels = (64,128,17)
            num_blocks = 3
            dilation_rate = 2
            final_conv_kernel = 3
            block = BasicBlock
        """
        super().__init__()
        self.bn_momentum = 0.1
        self.inp_channels = channels[0]
        self.num_joints_with_center = channels[
            2
        ]  # Should account for the center being a joint
        self.final_conv_kernel = final_conv_kernel

        self.transition_heatmap = self._make_transition_for_head(
            self.inp_channels, channels[1]
        )
        self.head_heatmap = self._make_heatmap_head(
            block, num_blocks, channels[1], dilation_rate
        )

    def _make_transition_for_head(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        """Summary:
        Construct the transition layer for the head.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels

        Returns:
            Transition layer consisting of Conv2d, BatchNorm2d, and ReLU
        """
        transition_layer = [
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(
        self,
        block: type(BaseBlock),
        num_blocks: int,
        num_channels: int,
        dilation_rate: int,
    ) -> nn.ModuleList:
        """Summary:
        Construct the heatmap head

        Args:
            block: type of block to use in the head.
            num_blocks: number of blocks in the head.
            num_channels: number of input channels for the head.
            dilation_rate: dilation rate for the head.

        Returns:
            List of modules representing the heatmap head layers.
        """
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            block, num_channels, num_channels, num_blocks, dilation=dilation_rate
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.num_joints_with_center,
            kernel_size=self.final_conv_kernel,
            stride=1,
            padding=1 if self.final_conv_kernel == 3 else 0,
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_layer(
        self,
        block: type(BaseBlock),
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        """Summary:
        Construct a layer in the head.

        Args:
            block: type of block to use in the head.
            in_channels: number of input channels for the layer.
            out_channels: number of output channels for the layer.
            num_blocks: number of blocks in the layer.
            stride: stride for the convolutional layer. Defaults to 1.
            dilation: dilation rate for the convolutional layer. Defaults to 1.

        Returns:
            Sequential layer containing the specified num_blocks.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels * block.expansion, momentum=self.bn_momentum
                ),
            )

        layers = [
            block(in_channels, out_channels, stride, downsample, dilation=dilation)
        ]
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        heatmap = self.head_heatmap[1](self.head_heatmap[0](self.transition_heatmap(x)))

        return heatmap


class DEKROffset(nn.Module):
    """
    DEKR module to compute the offset from the center corresponding to each keypoints:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang, CVPR 2021
    Code based on:
    https://github.com/HRNet/DEKR
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        num_offset_per_kpt: int,
        num_blocks: int,
        dilation_rate: int,
        final_conv_kernel: int,
        block: type(BaseBlock) = AdaptBlock,
    ) -> None:
        """Args:
        channels: tuple containing the number of input, offset, and output channels.
        num_offset_per_kpt: number of offset values per keypoint.
        num_blocks: number of blocks in the head.
        dilation_rate: dilation rate for convolutional layers.
        final_conv_kernel: kernel size for the final convolution.
        block: type of block to use in the head. Defaults to AdaptBlock.
        """
        super().__init__()
        self.inp_channels = channels[0]
        self.num_joints = channels[2]
        self.num_joints_with_center = self.num_joints + 1

        self.bn_momentum = 0.1
        self.offset_perkpt = num_offset_per_kpt
        self.num_joints_without_center = self.num_joints
        self.offset_channels = self.offset_perkpt * self.num_joints_without_center
        assert self.offset_channels == channels[1]

        self.num_blocks = num_blocks
        self.dilation_rate = dilation_rate
        self.final_conv_kernel = final_conv_kernel

        self.transition_offset = self._make_transition_for_head(
            self.inp_channels, self.offset_channels
        )
        (
            self.offset_feature_layers,
            self.offset_final_layer,
        ) = self._make_separete_regression_head(
            block,
            num_blocks=num_blocks,
            num_channels_per_kpt=self.offset_perkpt,
            dilation_rate=self.dilation_rate,
        )

    def _make_layer(
        self,
        block: type(BaseBlock),
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        """Summary:
        Create a sequential layer with the specified block and number of num_blocks.

        Args:
            block: block type to use in the layer.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_blocks: number of blocks to be stacked in the layer.
            stride: stride for the first block. Defaults to 1.
            dilation: dilation rate for the blocks. Defaults to 1.

        Returns:
            A sequential layer containing stacked num_blocks.

        Examples:
            input:
                block=BasicBlock
                in_channels=64
                out_channels=128
                num_blocks=3
                stride=1
                dilation=1
        """
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels * block.expansion, momentum=self.bn_momentum
                ),
            )

        layers = []
        layers.append(
            block(in_channels, out_channels, stride, downsample, dilation=dilation)
        )
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_for_head(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        """Summary:
        Create a transition layer for the head.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels

        Returns:
            Sequential layer containing the transition operations.
        """
        transition_layer = [
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ]
        return nn.Sequential(*transition_layer)

    def _make_separete_regression_head(
        self,
        block: type(BaseBlock),
        num_blocks: int,
        num_channels_per_kpt: int,
        dilation_rate: int,
    ) -> tuple:
        """Summary:

        Args:
            block: type of block to use in the head
            num_blocks: number of blocks in the regression head
            num_channels_per_kpt: number of channels per keypoint
            dilation_rate: dilation rate for the regression head

        Returns:
            A tuple containing two ModuleList objects.
            The first ModuleList contains the feature convolution layers for each keypoint,
            and the second ModuleList contains the final offset convolution layers.
        """
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                block,
                num_channels_per_kpt,
                num_channels_per_kpt,
                num_blocks,
                dilation=dilation_rate,
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=num_channels_per_kpt,
                out_channels=2,
                kernel_size=self.final_conv_kernel,
                stride=1,
                padding=1 if self.final_conv_kernel == 3 else 0,
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Summary:
        Perform forward pass through the OffsetDEKRHead.

        Args:
            x: input tensor to the head.

        Returns:
            offset: Computed offsets from the center corresponding to each keypoint.
            The tensor will have the shape (N, num_joints * 2, H, W), where N is the batch size,
            num_joints is the number of keypoints, and H and W are the height and width of the output tensor.
        """
        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[
                            :, j * self.offset_perkpt : (j + 1) * self.offset_perkpt
                        ]
                    )
                )
            )

        offset = torch.cat(final_offset, dim=1)

        return offset
