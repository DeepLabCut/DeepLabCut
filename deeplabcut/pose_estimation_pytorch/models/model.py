#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES
from deeplabcut.pose_estimation_pytorch.models.criterions import (
    CRITERIONS,
    LOSS_AGGREGATORS,
)
from deeplabcut.pose_estimation_pytorch.models.heads import HEADS, BaseHead
from deeplabcut.pose_estimation_pytorch.models.necks import NECKS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
)


class PoseModel(nn.Module):
    """A pose estimation model

    A pose estimation model is composed of a backbone, optionally a neck, and an
    arbitrary number of heads. Outputs are computed as follows:
    """

    def __init__(
        self,
        cfg: dict,
        backbone: torch.nn.Module,
        heads: dict[str, BaseHead],
        neck: torch.nn.Module = None,
        stride: int = 8,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
            stride: stride used in the model. Defaults to 8.
        """
        super().__init__()
        self.cfg = cfg

        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck
        self.stride = stride

        # TODO: Explore results, check batch size impact
        self.backbone.activate_batch_norm(False)

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)

        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_loss(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        total_losses = []
        losses: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            head_losses = head.get_loss(outputs[name], targets[name])
            total_losses.append(head_losses["total_loss"])
            for k, v in head_losses.items():
                losses[f"{name}_{k}"] = v

        # TODO: Different aggregation for multi-head loss?
        losses["total_loss"] = torch.mean(torch.stack(total_losses))
        return losses

    def get_target(
        self,
        inputs: torch.Tensor,
        outputs: dict[str, dict[str, torch.Tensor]],
        labels: dict,
    ) -> dict[str, dict]:
        """Summary:
        Get targets for model training.

        Args:
            inputs: the input images given to the model, of shape (b, c, w, h)
            outputs: output of each head group
            labels: dictionary of labels

        Returns:
            targets: dict of the targets for each model head group
        """
        return {
            name: head.target_generator(inputs, outputs[name], labels)
            for name, head in self.heads.items()
        }

    def get_predictions(
        self,
        inputs: torch.Tensor,
        outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            inputs: the input images given to the model, of shape (b, c, w, h)
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        return {
            head_name: head.predictor(inputs, outputs[head_name])
            for head_name, head in self.heads.items()
        }

    @staticmethod
    def from_cfg(cfg: dict) -> "PoseModel":
        backbone = BACKBONES.build(dict(cfg["backbone"]))

        neck = None
        if cfg.get("neck"):
            neck = NECKS.build(dict(cfg["neck"]))

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)
            if "type" in head_cfg["criterion"]:
                head_cfg["criterion"] = CRITERIONS.build(head_cfg["criterion"])
            else:
                weights = {}
                criterions = {}
                for loss_name, criterion_cfg in head_cfg["criterion"].items():
                    weights[loss_name] = criterion_cfg.get("weight", 1.0)
                    criterion_cfg = {
                        k: v for k, v in criterion_cfg.items() if k != "weight"
                    }
                    criterions[loss_name] = CRITERIONS.build(criterion_cfg)

                aggregator_cfg = {"type": "WeightedLossAggregator", "weights": weights}
                head_cfg["aggregator"] = LOSS_AGGREGATORS.build(aggregator_cfg)
                head_cfg["criterion"] = criterions

            head_cfg["target_generator"] = TARGET_GENERATORS.build(
                head_cfg["target_generator"]
            )
            head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
            heads[name] = HEADS.build(head_cfg)

        return PoseModel(
            cfg=cfg,
            backbone=backbone,
            neck=neck,
            heads=heads,
            **cfg["pose_model"],
        )
