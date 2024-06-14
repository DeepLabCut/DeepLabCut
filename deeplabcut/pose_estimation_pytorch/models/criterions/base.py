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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

LOSS_AGGREGATORS = Registry("loss_aggregators", build_func=build_from_cfg)
CRITERIONS = Registry("criterions", build_func=build_from_cfg)


class BaseCriterion(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            output: the output from which to compute the loss
            target: the target for the loss

        Returns:
            the different losses for the module, including one "total_loss" key which
            is the loss from which to start backpropagation
        """
        raise NotImplementedError


class BaseLossAggregator(ABC, nn.Module):
    @abstractmethod
    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: the losses to aggregate

        Returns:
            the aggregate loss
        """
        raise NotImplementedError
