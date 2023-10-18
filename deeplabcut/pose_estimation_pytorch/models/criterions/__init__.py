from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    LOSS_AGGREGATORS,
    CRITERIONS,
    BaseLossAggregator,
    BaseCriterion,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.aggregators import (
    WeightedLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.weighted import (
    WeightedBCECriterion,
    WeightedHuberCriterion,
    WeightedMSECriterion,
)
