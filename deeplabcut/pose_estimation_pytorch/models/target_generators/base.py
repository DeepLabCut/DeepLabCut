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
from abc import ABC, abstractmethod

import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

TARGET_GENERATORS = Registry("target_generators", build_func=build_from_cfg)


class BaseGenerator(ABC, nn.Module):
    """
    Given the ground truth annotation generates the corresponding maps for training the model
    """

    def __init__(self):
        super().__init__()
        self.batch_norm_on = False

    @abstractmethod
    def forward(self, x):
        pass
