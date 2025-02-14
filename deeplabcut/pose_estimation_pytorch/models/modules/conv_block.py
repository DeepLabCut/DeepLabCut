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
"""The code is based on DEKR: https://github.com/HRNet/DEKR/tree/main"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.ops as ops

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

BLOCKS = Registry("blocks", build_func=build_from_cfg)


class BaseBlock(ABC, nn.Module):
    """Abstract Base class for defining custom blocks.

    This class defines an abstract base class for creating custom blocks used in the HigherHRNet for Human Pose Estimation.

    Attributes:
        bn_momentum: Batch normalization momentum.

    Methods:
        forward(x): Abstract method for defining the forward pass of the block.
    """

    def __init__(self):
        super().__init__()
        self.bn_momentum = 0.1

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Abstract method for defining the forward pass of the block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        pass

    def _init_weights(self, pretrained: str | None):
        """Method for initializing block weights from pretrained models.

        Args:
            pretrained: Path to pretrained model weights.
        """
        if pretrained:
            self.load_state_dict(torch.load(pretrained))


@BLOCKS.register_module
class BasicBlock(BaseBlock):
    """Basic Residual Block.

    This class defines a basic residual block used in HigherHRNet.

    Attributes:
        expansion: The expansion factor used in the block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride value for the convolutional layers. Default is 1.
        downsample: Downsample layer to be used in the residual connection. Default is None.
        dilation: Dilation rate for the convolutional layers. Default is 1.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BasicBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BLOCKS.register_module
class Bottleneck(BaseBlock):
    """Bottleneck Residual Block.

    This class defines a bottleneck residual block used in HigherHRNet.

    Attributes:
        expansion: The expansion factor used in the block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride value for the convolutional layers. Default is 1.
        downsample: Downsample layer to be used in the residual connection. Default is None.
        dilation: Dilation rate for the convolutional layers. Default is 1.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            out_channels * self.expansion, momentum=self.bn_momentum
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Bottleneck block.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BLOCKS.register_module
class AdaptBlock(BaseBlock):
    """Adaptive Residual Block with Deformable Convolution.

    This class defines an adaptive residual block with deformable convolution used in HigherHRNet.

    Attributes:
        expansion: The expansion factor used in the block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride value for the convolutional layers. Default is 1.
        downsample: Downsample layer to be used in the residual connection. Default is None.
        dilation: Dilation rate for the convolutional layers. Default is 1.
        deformable_groups: Number of deformable groups in the deformable convolution. Default is 1.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
        deformable_groups: int = 1,
    ):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor(
            [[-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]]
        )
        self.register_buffer("regular_matrix", regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv2d(in_channels, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(in_channels, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            groups=deformable_groups,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AdaptBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape(
            (N * H * W, 2, 2)
        )
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset - self.regular_matrix
        offset = offset.transpose(1, 2).reshape((N, H, W, 18)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        offset[:, 1::2, :, :] += translation[:, 1:2, :, :]

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
