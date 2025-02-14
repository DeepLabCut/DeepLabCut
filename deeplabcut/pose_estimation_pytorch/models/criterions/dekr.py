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
"""Loss criterions for DEKR models"""
from __future__ import annotations

import torch

from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    BaseCriterion,
    CRITERIONS,
)


@CRITERIONS.register_module
class DEKRHeatmapLoss(BaseCriterion):
    """DEKR Heatmap loss"""

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: the output from which to compute the loss
            target: the target for the loss
            weights: the weights for the loss

        Returns:
            the DEKR offset loss
        """
        assert output.size() == target.size()
        loss = ((output - target) ** 2) * weights
        return loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)


@CRITERIONS.register_module
class DEKROffsetLoss(BaseCriterion):
    """DEKR Offset loss"""

    def __init__(self, beta: float = 1 / 9):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        return torch.where(
            l1_loss < self.beta,
            0.5 * l1_loss ** 2 / self.beta,
            l1_loss - 0.5 * self.beta,
        )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            output: the output from which to compute the loss
            target: the target for the loss
            weights: the weights for the loss

        Returns:
            the DEKR offset loss
        """
        assert output.size() == target.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(output, target) * weights
        if num_pos == 0:
            num_pos = 1.0
        loss = loss.sum() / num_pos
        return loss
