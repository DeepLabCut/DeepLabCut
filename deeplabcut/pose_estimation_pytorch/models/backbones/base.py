from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


BACKBONES = Registry('backbones', build_func=build_from_cfg)

class BaseBackbone(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm_on = False

    @abstractmethod
    def forward(self, x):
        pass

    def _init_weights(self, pretrained: str = None):
        """

        Parameters
        ----------
        pretrained

        Returns
        -------

        """
        if not pretrained:
            pass
        elif pretrained.startswith("http") or pretrained.startswith("ftp"):
            state_dict = torch.hub.load_state_dict_from_url(pretrained)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(torch.load(pretrained), strict=False)

    def activate_batch_norm(self, activation: bool=False):
        """Turns on or off batch norm layers updating their weights while training
        
        Prameters
        ---------
        activation:  should batch_norm be activated or not for training"""
        self.batch_norm_on = activation

    def train(self, mode = True):
        super(BaseBackbone, self).train(mode)

        if not self.batch_norm_on:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        return
