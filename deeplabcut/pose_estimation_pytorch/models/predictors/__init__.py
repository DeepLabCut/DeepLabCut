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
from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    PREDICTORS,
    BasePredictor,
)
from deeplabcut.pose_estimation_pytorch.models.predictors.dekr_predictor import (
    DEKRPredictor,
)
from deeplabcut.pose_estimation_pytorch.models.predictors.paf_predictor import (
    PartAffinityFieldPredictor,
)
from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
    HeatmapPredictor,
)
