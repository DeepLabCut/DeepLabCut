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
"""Implementation of modules needed for the CSPNeXt Backbone. Used in CSP-style models.

Based on the building blocks used for the ``mmdetection`` CSPNeXt implementation. For
more information, see <https://github.com/open-mmlab/mmdetection>.
"""
import torch
import torch.nn as nn


def build_activation(activation_fn: str, *args, **kwargs) -> nn.Module:
    if activation_fn == "SiLU":
        return nn.SiLU(*args, **kwargs)
    elif activation_fn == "ReLU":
        return nn.ReLU(*args, **kwargs)

    raise NotImplementedError(
        f"Unknown `CSPNeXT` activation: {activation_fn}. Must be one of 'SiLU', 'ReLU'"
    )


def build_norm(norm: str, *args, **kwargs) -> nn.Module:
    if norm == "SyncBN":
        return nn.SyncBatchNorm(*args, **kwargs)
    elif norm == "BN":
        return nn.BatchNorm2d(*args, **kwargs)

    raise NotImplementedError(
        f"Unknown `CSPNeXT` norm_layer: {norm}. Must be one of 'SyncBN', 'BN'"
    )


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP and (among others) CSPNeXt

    Args:
        in_channels: input channels to the bottleneck
        out_channels: output channels of the bottleneck
        kernel_sizes: kernel sizes for the pooling layers
        norm_layer: norm layer for the bottleneck
        activation_fn: activation function for the bottleneck
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        norm_layer: str | None = "SyncBN",
        activation_fn: str | None = "SiLU",
    ):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = CSPConvModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )

        self.poolings = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = CSPConvModule(
            conv2_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )

    def forward(self, x):
        x = self.conv1(x)
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels: Number of input/output channels of the layer.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPConvModule(nn.Module):
    """Configurable convolution module used for CSPNeXT.

    Applies sequentially
      - a convolution
      - (optional) a norm layer
      - (optional) an activation function

    Args:
        in_channels: Input channels of the convolution.
        out_channels: Output channels of the convolution.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Number of blocked connections from input to output channels.
        norm_layer: Norm layer to apply, if any.
        activation_fn: Activation function to apply, if any.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        norm_layer: str | None = None,
        activation_fn: str | None = "ReLU",
    ):
        super().__init__()

        self.with_activation = activation_fn is not None
        self.with_bias = norm_layer is None
        self.with_norm = norm_layer is not None

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=self.with_bias,
        )
        self.activate = None
        self.norm = None

        if self.with_norm:
            self.norm = build_norm(norm_layer, out_channels)

        if self.with_activation:
            # Careful when adding activation functions: some should not be in-place
            self.activate = build_activation(activation_fn, inplace=True)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        return x

    def _init_weights(self) -> None:
        """Same init as in <mmcv> convolutions"""
        nn.init.kaiming_normal_(self.conv.weight, a=0, nonlinearity="relu")
        if self.with_bias:
            nn.init.constant_(self.conv.bias, 0)

        if self.with_norm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)


class DepthwiseSeparableConv(nn.Module):
    """Depth-wise separable convolution module used for CSPNeXT.

    Applies sequentially
      - a depth-wise conv
      - a point-wise conv

    Args:
        in_channels: Input channels of the convolution.
        out_channels: Output channels of the convolution.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        norm_layer: Norm layer to apply, if any.
        activation_fn: Activation function to apply, if any.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        norm_layer: str | None = None,
        activation_fn: str | None = "ReLU",
    ):
        super().__init__()

        # depthwise convolution
        self.depthwise_conv = CSPConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )

        self.pointwise_conv = CSPConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class CSPNeXtBlock(nn.Module):
    """Basic bottleneck block used in CSPNeXt.

    Args:
        in_channels: input channels for the block
        out_channels: output channels for the block
        expansion: expansion factor for the hidden channels
        add_identity: add a skip-connection to the block
        kernel_size: kernel size for the DepthwiseSeparableConv
        norm_layer: Norm layer to apply, if any.
        activation_fn: Activation function to apply, if any.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        kernel_size: int = 5,
        norm_layer: str | None = None,
        activation_fn: str | None = "ReLU",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = CSPConvModule(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )
        self.conv2 = DepthwiseSeparableConv(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels: input channels for the layer
        out_channels: output channels for the block
        expand_ratio: expansion factor for the mid-channels
        num_blocks: the number of blocks to use
        add_identity: add a skip-connection to the blocks
        channel_attention: whether to apply channel attention
        norm_layer: Norm layer to apply, if any.
        activation_fn: Activation function to apply, if any.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        channel_attention: bool = False,
        norm_layer: str | None = None,
        activation_fn: str | None = "ReLU",
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = CSPConvModule(
            in_channels,
            mid_channels,
            1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )
        self.short_conv = CSPConvModule(
            in_channels,
            mid_channels,
            1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )
        self.final_conv = CSPConvModule(
            2 * mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer,
            activation_fn=activation_fn,
        )

        self.blocks = nn.Sequential(
            *[
                CSPNeXtBlock(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    norm_layer=norm_layer,
                    activation_fn=activation_fn,
                )
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)
