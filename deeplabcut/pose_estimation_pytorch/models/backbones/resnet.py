import torch.nn as nn
import torchvision

from deeplabcut.pose_estimation_pytorch.models.backbones.base import BaseBackbone, BACKBONES

@BACKBONES.register_module
class ResNet(BaseBackbone):

    def __init__(self, model_name: str = 'resnet50',
                 pretrained: str = None) -> nn.Module:
        """
        Parameters
        ----------
        model_name
        """
        super().__init__()
        _backbone = torchvision.models.get_model(model_name)
        _backbone._modules.pop('fc')
        _backbone._modules.pop('avgpool')
        self.model = nn.Sequential(_backbone._modules)
        self._init_weights(pretrained)

    def forward(self, x):
        return self.model(x)
