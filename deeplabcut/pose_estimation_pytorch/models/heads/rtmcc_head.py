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
"""Simple Coordinate Classification Head Implementation

From the paper "SimCC: a Simple Coordinate Classification Perspective for Human Pose
Estimation".
"""
from __future__ import annotations

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.heads.base import (
    BaseHead,
    HEADS,
)
from deeplabcut.pose_estimation_pytorch.models.modules import (
    GatedAttentionUnit,
    ScaleNorm,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.models.weight_init import BaseWeightInitializer


@HEADS.register_module
class RTMCCHead(BaseHead):
    """RTMPose Coordinate Classification head

    TODO: github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/rtmcc_block.py#L82
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        in_featuremap_size: tuple[int, int],
        simcc_split_ratio: float,
        final_layer_kernel_size: int,
        gau_cfg: dict,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion] | BaseCriterion,
        aggregator: BaseLossAggregator | None,
        weight_init: str | dict | BaseWeightInitializer | None = None,
    ) -> None:
        super().__init__(
            1,
            predictor,
            target_generator,
            criterion,
            aggregator,
            weight_init,
        )

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # from https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/projects/rtmpose/rtmpose/animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py#L85C9-L85C74
        # input size divided by scale
        # in_featuremap_size = tuple([s // 16 for s in input_size]),

        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        out_w = int(self.input_size[0] * self.simcc_split_ratio)
        out_h = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = GatedAttentionUnit(
            num_token=self.out_channels,
            in_token_dims=gau_cfg["hidden_dims"],
            out_token_dims=gau_cfg["hidden_dims"],
            expansion_factor=gau_cfg["expansion_factor"],
            s=gau_cfg["s"],
            eps=1e-5,
            dropout_rate=gau_cfg["dropout_rate"],
            drop_path=gau_cfg["drop_path"],
            attn_type="self-attn",
            act_fn=gau_cfg["act_fn"],
            use_rel_bias=gau_cfg["use_rel_bias"],
            pos_enc=gau_cfg["pos_enc"],
        )

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
        )
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg["hidden_dims"], bias=False),
        )

        self.cls_x = nn.Linear(gau_cfg["hidden_dims"], out_w, bias=False)
        self.cls_y = nn.Linear(gau_cfg["hidden_dims"], out_h, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.final_layer(x)  # -> B, K, H, W
        feats = torch.flatten(feats, 2)  # -> B, K, hidden
        feats = self.mlp(feats)  # -> B, K, hidden
        feats = self.gau(feats)
        x, y = self.cls_x(feats), self.cls_y(feats)
        return dict(x=x, y=y)
