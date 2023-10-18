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


@BACKBONES.register_module
class ResNet(BaseBackbone):
    """ResNet backbone.

    This class represents a typical ResNet backbone for pose estimation.

    Attributes:
        model: the ResNet model
    """

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True) -> None:
        """Initialize the ResNet backbone.

        Args:
            model_name: Name of the ResNet model to use, e.g., 'resnet50', 'resnet101'
            pretrained: If True, initializes with ImageNet pretrained weights.
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet backbone.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import ResNet
            >>> backbone = ResNet(model_name='resnet50', pretrained=False)
            >>> x = torch.randn(1, 3, 256, 256)
            >>> y = backbone(x)

            Expected Output Shape:
                If input size is (batch_size, 3, shape_x, shape_y), the output shape
                will be (batch_size, 3, shape_x//32, shape_y//32)
        """
        return self.model.forward_features(x)
