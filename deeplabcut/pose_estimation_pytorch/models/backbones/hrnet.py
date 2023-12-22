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

    Attributes:
        model: the HRNet model
    """

    def __init__(
        self,
        model_name: str = "hrnet_w32",
        pretrained: bool = True,
        only_high_res: bool = False,
    ) -> None:
        """Constructs an ImageNet pretrained HRNet from timm.

        Args:
            model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
            pretrained: If True, loads the model with ImageNet pretrained weights.
            only_high_res: Whether to only return the high resolution features
        """
        super().__init__()
        self.model = _load_hrnet(model_name, pretrained)
        self.only_high_res = only_high_res

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
        y_list = self.model.forward_features(x)
        if self.only_high_res:
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


def _load_hrnet(model_name: str, pretrained: bool) -> nn.Module:
    """
    Loads a TIMM HRNet model, while setting incre_modules to None. This is necessary to
    get high-resolution features; otherwise model.forward_features() returns
    low-resolution maps.

    Args:
        model_name: the name of the HRNet model to load
        pretrained: whether the ImageNet pretrained weights should be loaded

    Returns:
        the HRNet model
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    model.incre_modules = None
    return model
