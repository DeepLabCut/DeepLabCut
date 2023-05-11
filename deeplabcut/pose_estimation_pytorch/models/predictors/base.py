import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

PREDICTORS = Registry('predictors', build_func=build_from_cfg)

class BasePredictor(ABC, nn.Module):

    def __init__(self):
        super().__init__()

        self.num_animals = None

    @abstractmethod
    def forward(self, outputs):
        pass