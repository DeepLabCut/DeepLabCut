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
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import BaseCriterion
from deeplabcut.pose_estimation_pytorch.models.heads import BaseHead, HEADS
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator


@HEADS.register_module
class TransformerHead(BaseHead):
    """
    Transformer Head module to predict heatmaps using a transformer-based approach
    """

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: BaseCriterion,
        dim: int,
        hidden_heatmap_dim: int,
        heatmap_dim: int,
        apply_multi: bool,
        heatmap_size: tuple[int, int],
        apply_init: bool,
        head_stride: int,
    ):
        """
        Args:
            dim: Dimension of the input features.
            hidden_heatmap_dim: Dimension of the hidden features in the MLP head.
            heatmap_dim: Dimension of the output heatmaps.
            apply_multi: If True, apply a multi-layer perceptron (MLP) with LayerNorm
                to generate heatmaps. If False, directly apply a single linear
                layer for heatmap prediction.
            heatmap_size: Tuple (height, width) representing the size of the output
                heatmaps.
            apply_init: If True, apply weight initialization to the module's layers.
            head_stride: The stride for the head (or neck + head pair), where positive
                values indicate an increase in resolution while negative values a
                decrease. Assuming that H and W are divisible by head_stride, this is
                the value such that if a backbone outputs an encoding of shape
                (C, H, W), the head will output heatmaps of shape:
                    (C, H * head_stride, W * head_stride)    if head_stride > 0
                    (C, -H/head_stride, -W/head_stride)      if head_stride < 0
        """
        super().__init__(head_stride, predictor, target_generator, criterion)
        self.mlp_head = (
            nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, hidden_heatmap_dim),
                nn.LayerNorm(hidden_heatmap_dim),
                nn.Linear(hidden_heatmap_dim, heatmap_dim),
            )
            if (dim * 3 <= hidden_heatmap_dim * 0.5 and apply_multi)
            else nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, heatmap_dim))
        )
        self.heatmap_size = heatmap_size
        # trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.mlp_head(x)
        x = rearrange(
            x,
            "b c (p1 p2) -> b c p1 p2",
            p1=self.heatmap_size[0],
            p2=self.heatmap_size[1],
        )
        return {"heatmap": x}

    def _init_weights(self, m: nn.Module) -> None:
        """
        Custom weight initialization for linear and layer normalization layers.

        Args:
            m: module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
