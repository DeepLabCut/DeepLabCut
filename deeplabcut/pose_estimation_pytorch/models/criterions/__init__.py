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
from deeplabcut.pose_estimation_pytorch.models.criterions.aggregators import (
    WeightedLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    CRITERIONS,
    LOSS_AGGREGATORS,
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.dekr import (
    DEKRHeatmapLoss,
    DEKROffsetLoss,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.kl_discrete import (
    KLDiscreteLoss,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.weighted import (
    WeightedBCECriterion,
    WeightedHuberCriterion,
    WeightedMSECriterion,
)
