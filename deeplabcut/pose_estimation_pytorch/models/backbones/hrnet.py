import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BaseBackbone,
    BACKBONES,
)


@BACKBONES.register_module
class HRNet(BaseBackbone):
    """
    HRNet backbone, this version returns high resolution feature maps of size
    1/4 * original_image_size
    This is obtained using bilinear interpolation and concatenation of all the outputs of the
    HRNet stages
    """

    def __init__(self, model_name: str = "hrnet_w32") -> nn.Module:
        """
        Constructs an ImageNet pre-trained HRNet from timm
        (https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/hrnet.py)

        Parameters
        ----------
            model_name: str
                type of HRNet (e.g. 'hrnet_w32, 'hrnet_w48')
        """
        super().__init__()
        _backbone = timm.create_model(model_name, pretrained=True)
        _backbone.incre_modules = None  # Necesssary to get high resolution features, if not set to None _backbone.forward_features will return low_res images
        self.model = _backbone

    def forward(self, x):
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
    def __init__(self, model_name: str = "hrnet_w32"):
        """
        Constructs an ImageNet pre-trained HRNet from timm
        (https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/hrnet.py)

        Parameters
        ----------
            model_name: str
                type of HRNet (e.g. 'hrnet_w32, 'hrnet_w48')
        """
        super().__init__()
        _backbone = timm.create_model(model_name, pretrained=True)
        _backbone.incre_modules = None  # Necesssary to get high resolution features, if not set to None _backbone.forward_features will return low_res images
        self.model = _backbone

    def forward(self, x):
        return self.model.forward_features(x)[
            0
        ]  # Only take the high resolution stream at the end
