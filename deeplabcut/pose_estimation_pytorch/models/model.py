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

import copy
import logging

import torch
import torch.nn as nn

import deeplabcut.pose_estimation_pytorch.modelzoo.utils as modelzoo_utils
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone
from deeplabcut.pose_estimation_pytorch.models.criterions import (
    CRITERIONS,
    LOSS_AGGREGATORS,
)
from deeplabcut.pose_estimation_pytorch.models.heads import BaseHead, HEADS
from deeplabcut.pose_estimation_pytorch.models.necks import BaseNeck, NECKS
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
        backbone: BaseBackbone,
        heads: dict[str, BaseHead],
        neck: BaseNeck | None = None,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck

        self._strides = {
            name: _model_stride(self.backbone.stride, head.stride)
            for name, head in heads.items()
        }

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
        outputs: dict[str, dict[str, torch.Tensor]],
        labels: dict,
    ) -> dict[str, dict]:
        """Summary:
        Get targets for model training.

        Args:
            outputs: output of each head group
            labels: dictionary of labels

        Returns:
            targets: dict of the targets for each model head group
        """
        return {
            name: head.target_generator(self._strides[name], outputs[name], labels)
            for name, head in self.heads.items()
        }

    def get_predictions(self, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        return {
            name: head.predictor(self._strides[name], outputs[name])
            for name, head in self.heads.items()
        }

    @staticmethod
    def build(
        cfg: dict,
        weight_init: None | WeightInitialization = None,
        pretrained_backbone: bool = False,
    ) -> "PoseModel":
        """
        Args:
            cfg: The configuration of the model to build.
            weight_init: How model weights should be initialized. If None, ImageNet
                pre-trained backbone weights are loaded from Timm.
            pretrained_backbone: Whether to load an ImageNet-pretrained weights for
                the backbone. This should only be set to True when building a model
                which will be trained on a transfer learning task.

        Returns:
            the built pose model
        """
        cfg["backbone"]["pretrained"] = pretrained_backbone
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

        model = PoseModel(cfg=cfg, backbone=backbone, neck=neck, heads=heads)

        if weight_init is not None:
            logging.info(f"Loading pretrained model weights: {weight_init}")

            # TODO: Should we specify the pose_model_type in WeightInitialization?
            backbone_name = cfg["backbone"]["model_name"]
            pose_model_type = modelzoo_utils.get_pose_model_type(backbone_name)

            # load pretrained weights
            if weight_init.customized_pose_checkpoint is None:
                _, _, snapshot_path, _ = modelzoo_utils.get_config_model_paths(
                    project_name=weight_init.dataset,
                    pose_model_type=pose_model_type,
                )
            else:
                snapshot_path = weight_init.customized_pose_checkpoint

            logging.info(f"The pose model is loading from {snapshot_path}")
            snapshot = torch.load(snapshot_path, map_location="cpu")
            state_dict = snapshot["model"]

            # load backbone state dict
            model.backbone.load_state_dict(filter_state_dict(state_dict, "backbone"))

            # if there's a neck, load state dict
            if model.neck is not None:
                model.neck.load_state_dict(filter_state_dict(state_dict, "neck"))

            # load head state dicts
            if weight_init.with_decoder:
                all_head_state_dicts = filter_state_dict(state_dict, "heads")
                conversion_tensor = torch.from_numpy(weight_init.conversion_array)
                for name, head in model.heads.items():
                    head_state_dict = filter_state_dict(all_head_state_dicts, name)

                    # requires WeightConversionMixin
                    if not weight_init.memory_replay:
                        head_state_dict = head.convert_weights(
                            state_dict=head_state_dict,
                            module_prefix="",
                            conversion=conversion_tensor,
                        )

                    head.load_state_dict(head_state_dict)

        return model


def filter_state_dict(state_dict: dict, module: str) -> dict[str, torch.Tensor]:
    """
    Filters keys in the state dict for a module to only keep a given prefix. Removes
    the module from the keys (e.g. for module="backbone", "backbone.stage1.weight" will
    be converted to "stage1.weight" so the state dict can be loaded into the backbone
    directly).

    Args:
        state_dict: the state dict
        module: the module to keep, e.g. "backbone"

    Returns:
        the filtered state dict, with the module removed from the keys

    Examples:
        state_dict = {"backbone.conv.weight": t1, "head.conv.weight": t2}
        filtered = filter_state_dict(state_dict, "backbone")
        # filtered = {"conv.weight": t1}
        model.backbone.load_state_dict(filtered)
    """
    return {
        ".".join(k.split(".")[1:]): v  # remove 'backbone.' from the keys
        for k, v in state_dict.items()
        if k.startswith(module)
    }


def _model_stride(backbone_stride: int | float, head_stride: int | float) -> float:
    """Computes the model stride from a backbone and a head"""
    if head_stride > 0:
        return backbone_stride / head_stride

    return backbone_stride * -head_stride
