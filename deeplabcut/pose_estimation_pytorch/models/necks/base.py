from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

NECKS = Registry("necks", build_func=build_from_cfg)


class BaseNeck(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def _init_weights(self, pretrained):
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained))
