from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


TARGET_GENERATORS = Registry('target_generators', build_func=build_from_cfg)

class BaseGenerator(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_norm_on = False

    @abstractmethod
    def forward(self, x):
        pass

    