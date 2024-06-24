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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplabcut.pose_estimation_pytorch.models.criterions import utils
from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    BaseCriterion,
    CRITERIONS,
)


class WeightedCriterion(BaseCriterion):
    """Base class for weighted criterions"""

    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: predicted tensor
            target: target tensor
            weights: weights for each element in the loss calculation. If a float,
                weights all elements by that value. Defaults to 1.

        Returns:
            the weighted loss
        """
        # shape of loss: (batch_size, n_kpts, heatmap_size, heatmap_size)
        loss = self.criterion(output, target)
        n_elems = utils.count_nonzero_elems(loss, weights)
        if n_elems == 0:
            n_elems = 1

        return torch.sum(loss * weights) / n_elems


@CRITERIONS.register_module
class WeightedMSECriterion(WeightedCriterion):
    """
    Weighted Mean Squared Error (MSE) Loss.

    This loss computes the Mean Squared Error between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 (masked items)
    are excluded from the loss calculation.
    """

    def __init__(self) -> None:
        super().__init__(nn.MSELoss(reduction="none"))

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: predicted tensor
            target: target tensor
            weights: weights for each element in the loss calculation. If a float,
                weights all elements by that value. Defaults to 1.

        Returns:
            the weighted loss
        """
        # shape of loss: (batch_size, n_kpts, h, w)
        loss = self.criterion(output, target)
        n_elems = utils.count_nonzero_elems(loss, weights)
        if n_elems == 0:
            n_elems = 1

        return torch.sum(loss * weights) / n_elems


@CRITERIONS.register_module
class WeightedHuberCriterion(WeightedCriterion):
    """
    Weighted Huber Loss.

    This loss computes the Huber loss between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 are
    excluded from the loss calculation.
    """

    def __init__(self) -> None:
        super().__init__(nn.HuberLoss(reduction="none"))


@CRITERIONS.register_module
class WeightedBCECriterion(WeightedCriterion):
    """
    Weighted Binary Cross Entropy (BCE) Loss.

    This loss computes the Binary Cross Entropy loss between the prediction and target tensors,
    but it also incorporates weights to adjust the contribution of each element in the loss
    calculation. The loss is computed element-wise, and elements with a weight of 0 are
    excluded from the loss calculation.
    """

    def __init__(self) -> None:
        super().__init__(nn.BCEWithLogitsLoss(reduction="none"))
