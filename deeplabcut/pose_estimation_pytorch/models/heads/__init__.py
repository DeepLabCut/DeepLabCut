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
from deeplabcut.pose_estimation_pytorch.models.heads.base import HEADS, BaseHead
from deeplabcut.pose_estimation_pytorch.models.heads.dekr import DEKRHead
from deeplabcut.pose_estimation_pytorch.models.heads.dlcrnet import DLCRNetHead
from deeplabcut.pose_estimation_pytorch.models.heads.simple_head import HeatmapHead
from deeplabcut.pose_estimation_pytorch.models.heads.transformer import TransformerHead
