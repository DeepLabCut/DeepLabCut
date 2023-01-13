from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


BACKBONES = Registry('backbones', build_func=build_from_cfg)

class BaseBackbone(ABC, nn.Module):

    def __init__(self):
        super().__init__()

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
