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


PREDICTORS = Registry('predictors', build_func=build_from_cfg)


class BasePredictor(ABC, nn.Module):
    """ A base predictor """

    def __init__(self):
        super().__init__()

        self.num_animals = None

    @abstractmethod
    def forward(self, outputs):
        pass
