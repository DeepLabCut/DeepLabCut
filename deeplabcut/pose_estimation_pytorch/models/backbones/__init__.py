#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
from deeplabcut.pose_estimation_pytorch.models.backbones.hrnet import HRNet
from deeplabcut.pose_estimation_pytorch.models.backbones.resnet import ResNet, DLCRNet
