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
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.heads.base import HEADS
from deeplabcut.pose_estimation_pytorch.models.heads.simple_head import (
    DeconvModule,
    HeatmapHead,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.models.weight_init import BaseWeightInitializer


@HEADS.register_module
class DLCRNetHead(HeatmapHead):
    """A head for DLCRNet models using Part-Affinity Fields to predict individuals"""

    def __init__(
        self,
        predictor: BasePredictor,
        target_generator: BaseGenerator,
        criterion: dict[str, BaseCriterion],
        aggregator: BaseLossAggregator,
        heatmap_config: dict,
        locref_config: dict,
        paf_config: dict,
        num_stages: int = 5,
        features_dim: int = 128,
        weight_init: str | dict | BaseWeightInitializer | None = None,
    ) -> None:
        self.num_stages = num_stages
        # FIXME Cleaner __init__ to avoid initializing unused layers
        in_channels = heatmap_config["channels"][0]
        num_keypoints = heatmap_config["channels"][-1]
        num_limbs = paf_config["channels"][-1]  # Already has the 2x multiplier
        in_refined_channels = features_dim + num_keypoints + num_limbs
        if num_stages > 0:
            heatmap_config["channels"][0] = paf_config["channels"][0] = (
                in_refined_channels
            )
            locref_config["channels"][0] = locref_config["channels"][-1]
        super().__init__(
            predictor,
            target_generator,
            criterion,
            aggregator,
            heatmap_config,
            locref_config,
            weight_init,
        )
        self.paf_head = DeconvModule(**paf_config)

        self.convt1 = self._make_layer_same_padding(
            in_channels=in_channels, out_channels=num_keypoints
        )
        self.convt2 = self._make_layer_same_padding(
            in_channels=in_channels, out_channels=locref_config["channels"][-1]
        )
        self.convt3 = self._make_layer_same_padding(
            in_channels=in_channels, out_channels=num_limbs
        )
        self.convt4 = self._make_layer_same_padding(
            in_channels=in_channels, out_channels=features_dim
        )
        self.hm_ref_layers = nn.ModuleList()
        self.paf_ref_layers = nn.ModuleList()
        for _ in range(num_stages):
            self.hm_ref_layers.append(
                self._make_refinement_layer(
                    in_channels=in_refined_channels, out_channels=num_keypoints
                )
            )
            self.paf_ref_layers.append(
                self._make_refinement_layer(
                    in_channels=in_refined_channels, out_channels=num_limbs
                )
            )
        self._init_weights()

    def _make_layer_same_padding(
        self, in_channels: int, out_channels: int
    ) -> nn.ConvTranspose2d:
        # FIXME There is no consensual solution to emulate TF behavior in pytorch
        # see https://github.com/pytorch/pytorch/issues/3867
        return nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def _make_refinement_layer(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        """Summary:
        Helper function to create a refinement layer.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels

        Returns:
            refinement_layer: the refinement layer.
        """
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding="same"
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.num_stages > 0:
            stage1_hm_out = self.convt1(x)
            stage1_paf_out = self.convt3(x)
            features = self.convt4(x)
            stage2_in = torch.cat((stage1_hm_out, stage1_paf_out, features), dim=1)
            stage_in = stage2_in
            stage_paf_out = stage1_paf_out
            stage_hm_out = stage1_hm_out
            for i, (hm_ref_layer, paf_ref_layer) in enumerate(
                zip(self.hm_ref_layers, self.paf_ref_layers)
            ):
                pre_stage_hm_out = stage_hm_out
                stage_hm_out = hm_ref_layer(stage_in)
                stage_paf_out = paf_ref_layer(stage_in)
                if i > 0:
                    stage_hm_out += pre_stage_hm_out
                stage_in = torch.cat((stage_hm_out, stage_paf_out, features), dim=1)
            return {
                "heatmap": self.heatmap_head(stage_in),
                "locref": self.locref_head(self.convt2(x)),
                "paf": self.paf_head(stage_in),
            }
        return {
            "heatmap": self.heatmap_head(x),
            "locref": self.locref_head(x),
            "paf": self.paf_head(x),
        }
