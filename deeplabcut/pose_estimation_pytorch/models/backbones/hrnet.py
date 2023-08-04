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
from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
from torch.nn import functional as F


@BACKBONES.register_module
class HRNet(BaseBackbone):
    """HRNet backbone.

    This version returns high-resolution feature maps of size 1/4 * original_image_size.
    This is obtained using bilinear interpolation and concatenation of all the outputs
    of the HRNet stages.

    Args:
        model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
        pretrained: If True, loads the model with ImageNet pre-trained weights.
    """

    def __init__(
        self, model_name: str = "hrnet_w32", pretrained: bool = True
    ) -> torch.nn.Module:
        """Constructs an ImageNet pre-trained HRNet from timm.

        Args:
            model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
            pretrained: If True, loads the model with ImageNet pre-trained weights.
        """
        super().__init__()
        _backbone = timm.create_model(model_name, pretrained=pretrained)
        _backbone.incre_modules = None  # Necessary to get high-resolution features; otherwise, _backbone.forward_features will return low-resolution images.
        self.model = _backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HRNet backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor with concatenated high-resolution feature maps.

        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import HRNet
            >>> backbone = HRNet(model_name='hrnet_w32', pretrained=False)
            >>> x = torch.randn(1, 3, 256, 256)
            >>> y = backbone(x)
        """
        y_list = self.model.forward_features(x)
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


@BACKBONES.register_module
class HRNetTopDown(BaseBackbone):
    """HRNet backbone for the top-down approach.
    This version returns only the high-resolution stream.

    Args:
        model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
        pretrained: If True, loads the model with ImageNet pre-trained weights.
    """

    def __init__(self, model_name: str = "hrnet_w32", pretrained: bool = True):
        """Constructs an ImageNet pre-trained HRNet from timm.

        Args:
            model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
            pretrained: If True, loads the model with ImageNet pre-trained weights.
        """
        super().__init__()
        _backbone = timm.create_model(model_name, pretrained=pretrained)
        _backbone.incre_modules = None  # Necessary to get high-resolution features; otherwise, _backbone.forward_features will return low-resolution images.
        self.model = _backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HRNet backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Output tensor with the high-resolution stream.

        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import HRNetTopDown
            >>> backbone = HRNetTopDown(model_name='hrnet_w32', pretrained=False)
            >>> x = torch.randn(1, 3, 256, 256)
            >>> y = backbone(x)
        """
        return self.model.forward_features(x)[
            0
        ]  # Only take the high-resolution stream at the end
