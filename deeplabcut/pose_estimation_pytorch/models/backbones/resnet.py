import torch.nn as nn
import timm
from typing import Union

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BaseBackbone,
    BACKBONES,
)


@BACKBONES.register_module
class ResNet(BaseBackbone):
    """
    Typical ResNet backbone
    """

    def __init__(
        self, model_name: str = "resnet50", pretrained: bool = True
    ) -> nn.Module:
        """
        Parameters
        ----------
        model_name
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x):
        return self.model.forward_features(x)
