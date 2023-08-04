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
from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
import torch.nn as nn


@BACKBONES.register_module
class ResNet(BaseBackbone):
    """ResNet backbone.

    This class represents a typical ResNet backbone for pose estimation.

    Args:
        model_name: Name of the ResNet model to use, e.g., 'resnet50', 'resnet101', etc.
                                   Defaults to 'resnet50'.
        pretrained: If True, the backbone will be initialized with ImageNet pre-trained weights.
                                     Defaults to True.
    """

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True) -> None:
        """Initialize the ResNet backbone.

        Args:
            model_name: Name of the ResNet model to use, e.g., 'resnet50', 'resnet101', etc.
                                       Defaults to 'resnet50'.
            pretrained: If True, the backbone will be initialized with ImageNet pre-trained weights.
                                         Defaults to True.
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x):
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
                If input size is (batch_size, 3, shape_x, shape_y), the output shape will be (batch_size, 3, shape_x//32, shape_y//32)
        """
        return self.model.forward_features(x)
