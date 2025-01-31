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
from __future__ import annotations

import torch

from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    BaseLossAggregator,
    LOSS_AGGREGATORS,
)


@LOSS_AGGREGATORS.register_module
class WeightedLossAggregator(BaseLossAggregator):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weighted_losses = [
            weight * losses[loss_name] for loss_name, weight in self.weights.items()
        ]
        return torch.mean(torch.stack(weighted_losses))
