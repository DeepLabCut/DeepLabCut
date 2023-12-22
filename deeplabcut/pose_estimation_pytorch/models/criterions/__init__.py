from deeplabcut.pose_estimation_pytorch.models.criterions.aggregators import (
    WeightedLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    CRITERIONS,
    LOSS_AGGREGATORS,
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.criterions.weighted import (
    WeightedBCECriterion,
    WeightedHuberCriterion,
    WeightedMSECriterion,
)
