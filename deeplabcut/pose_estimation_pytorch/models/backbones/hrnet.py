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
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)


@BACKBONES.register_module
class HRNet(BaseBackbone):
    """HRNet backbone.

    This version returns high-resolution feature maps of size 1/4 * original_image_size.
    This is obtained using bilinear interpolation and concatenation of all the outputs
    of the HRNet stages.

    The model outputs 4 branches, with strides 4, 8, 16 and 32.

    Args:
        stride: The stride of the HRNet. Should always be 4, except for custom models.
        model_name: Any HRNet variant available through timm (e.g., 'hrnet_w32',
            'hrnet_w48'). See timm for more options.
        pretrained: If True, loads the backbone with ImageNet pretrained weights from
            timm.
        interpolate_branches: Needed for DEKR. Instead of returning features from the
            high-resolution branch, interpolates all other branches to the same shape
            and concatenates them.
        increased_channel_count: As described by timm, it "allows grabbing increased
            channel count features using part of the classification head" (otherwise,
            the default features are returned).
        kwargs: BaseBackbone kwargs

    Attributes:
        model: the HRNet model
    """

    def __init__(
        self,
        stride: int = 4,
        model_name: str = "hrnet_w32",
        pretrained: bool = False,
        interpolate_branches: bool = False,
        increased_channel_count: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(stride=stride, **kwargs)
        self.model = _load_hrnet(model_name, pretrained, increased_channel_count)
        self.interpolate_branches = interpolate_branches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HRNet backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            the feature map

        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import HRNet
            >>> backbone = HRNet(model_name='hrnet_w32', pretrained=False)
            >>> x = torch.randn(1, 3, 256, 256)
            >>> y = backbone(x)
        """
        y_list = self.model(x)
        if not self.interpolate_branches:
            return y_list[0]

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat(
            [
                y_list[0],
                F.interpolate(y_list[1], size=(x0_h, x0_w), mode="bilinear"),
                F.interpolate(y_list[2], size=(x0_h, x0_w), mode="bilinear"),
                F.interpolate(y_list[3], size=(x0_h, x0_w), mode="bilinear"),
            ],
            1,
        )
        return x


def _load_hrnet(
    model_name: str,
    pretrained: bool,
    increased_channel_count: bool,
) -> nn.Module:
    """Loads a TIMM HRNet model.

    Args:
        model_name: Any HRNet variant available through timm (e.g., 'hrnet_w32',
            'hrnet_w48'). See timm for more options.
        pretrained: If True, loads the backbone with ImageNet pretrained weights from
            timm.
        increased_channel_count: As described by timm, it "allows grabbing increased
            channel count features using part of the classification head" (otherwise,
            the default features are returned).

    Returns:
        the HRNet model
    """
    # First stem conv is used for stride 2 features, so only return branches 1-4
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
        feature_location="incre" if increased_channel_count else "",
        out_indices=(1, 2, 3, 4),
    )
