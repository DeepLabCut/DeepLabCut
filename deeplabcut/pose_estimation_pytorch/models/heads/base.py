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
from deeplabcut.pose_estimation_pytorch.models.weight_init import (
    BaseWeightInitializer,
    WEIGHT_INIT,
)
from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry

HEADS = Registry("heads", build_func=build_from_cfg)


class BaseHead(ABC, nn.Module):
    """A head for pose estimation models

    Attributes:
        stride: The stride for the head (or neck + head pair), where positive values
            indicate an increase in resolution while negative values a decrease.
            Assuming that H and W are divisible by `stride`, this is the value such
            that if a backbone outputs an encoding of shape (C, H, W), the head will
            output heatmaps of shape:
                (C, H * stride, W * stride)       if stride > 0
                (C, -H/stride, -W/stride)         if stride < 0
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
        stride: int | float,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion] | BaseCriterion,
        aggregator: BaseLossAggregator | None = None,
        weight_init: str | dict | BaseWeightInitializer | None = None,
    ) -> None:
        super().__init__()
        if stride == 0:
            raise ValueError(f"Stride must not be 0. Found {stride}.")

        self.stride = stride
        self.predictor = predictor
        self.target_generator = target_generator
        self.criterion = criterion
        self.aggregator = aggregator

        self.weight_init: BaseWeightInitializer | None = None
        if isinstance(weight_init, BaseWeightInitializer):
            self.weight_init = weight_init
        elif isinstance(weight_init, (str, dict)):
            self.weight_init = WEIGHT_INIT.build(weight_init)
        elif weight_init is not None:
            raise ValueError(
                f"Could not parse ``weight_init`` parameter: {weight_init}."
            )

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

    def _init_weights(self) -> None:
        """Should be called once all modules for the class are created"""
        if self.weight_init is not None:
            self.weight_init.init_weights(self)


class WeightConversionMixin(ABC):
    """A mixin for heads that can re-order and/or filter the output channels.

    This mixin is useful to convert SuperAnimal model weights such that they can be used
    in downstream projects (either existing or new), where only a subset of keypoints
    are available (and where they might be re-ordered).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def convert_weights(
        state_dict: dict[str, torch.Tensor],
        module_prefix: str,
        conversion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Converts pre-trained weights to be fine-tuned on another dataset

        Args:
            state_dict: the state dict for the pre-trained model
            module_prefix: the prefix for weights in this head (e.g., 'heads.bodypart.')
            conversion: the mapping of old indices to new indices

        Examples:
            A SuperAnimal model was trained on the keypoints ["ear_left", "ear_right",
            "eye_left", "eye_right", "nose"]. A down-stream project has the bodyparts
            labeled ["nose", "eye_left", "eye_right"]. The SuperAnimal weights can be
            converted (to be used with the downstream project) with the following code:

                ``
                state_dict = torch.load(
                    snapshot_path, map_location=torch.device('cpu')
                )["model"]
                state_dict = HeadClass.convert_weights(
                    state_dict,
                    "heads.bodypart",
                    [4, 2, 3]
                )
                ``
        """
        pass
