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

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

HEADS = Registry("heads", build_func=build_from_cfg)


class BaseHead(ABC, nn.Module):
    """A head for pose estimation models

    Attributes:
        predictor: an object to generate predictions from the head outputs
        target_generator: a target generator which must output a target for each
            output key of this module (i.e. if forward returns a "heatmap" tensor and
            an "offset" tensor, then targets must be generated for both)
        criterion: either a single criterion (e.g. if this head only outputs heatmaps)
            or a dictionary mapping the outputs of this head to the criterion to use
            (e.g. a "heatmap" criterion and an "offset" criterion for DEKR).
        aggregator: if the criterion is a dictionary, cannot be none. used to combine
            the individual losses from this head into one "total_loss"
    """

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion] | BaseCriterion,
        aggregator: BaseLossAggregator | None = None,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.target_generator = target_generator
        self.criterion = criterion
        self.aggregator = aggregator

        if isinstance(criterion, dict):
            if aggregator is None:
                raise ValueError(
                    f"When multiple criterions are defined, a loss aggregator must "
                    "also be given"
                )
        else:
            if aggregator is not None:
                raise ValueError(
                    f"Cannot use a loss aggregator with a single criterion"
                )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Given the feature maps for an image ()

        Args:
            x: the feature maps, of shape (b, c, h, w)

        Returns:
            the head outputs (e.g. "heatmap", "locref")
        """
        pass

    def get_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Computes the loss for this head

        Args:
            outputs: the outputs of this head
            targets: the targets for this head

        Returns:
            A dictionary containing minimally "total_loss" key mapping to the total
            loss for this head (from which backwards() should be called). Can contain
            other keys containing losses that can be logged for informational purposes.
        """
        if self.aggregator is None:
            assert len(outputs) == len(targets) == 1
            key = [k for k in outputs.keys()][0]
            return {"total_loss": self.criterion(outputs[key], **targets[key])}

        losses = {
            name: criterion(outputs[name], **targets[name])
            for name, criterion in self.criterion.items()
        }
        losses["total_loss"] = self.aggregator(losses)
        return losses
