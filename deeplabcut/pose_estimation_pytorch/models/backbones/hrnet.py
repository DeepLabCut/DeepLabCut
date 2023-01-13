import timm
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import BaseBackbone, BACKBONES

@BACKBONES.register_module
class HRNet(BaseBackbone):

    def __init__(self, model_name: str = 'hrnet_w32') -> nn.Module:
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
        self.model = _backbone

    def forward(self, x):
        return self.model.forward_features(x)
