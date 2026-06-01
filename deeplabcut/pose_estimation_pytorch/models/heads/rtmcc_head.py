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
"""Modified SimCC head for the RTMPose model.

Based on the official ``mmpose`` RTMCC head implementation. For more information, see
<https://github.com/open-mmlab/mmpose>.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.criterions import (
    BaseCriterion,
    BaseLossAggregator,
)
from deeplabcut.pose_estimation_pytorch.models.heads.base import (
    HEADS,
    BaseHead,
    WeightConversionMixin,
)
from deeplabcut.pose_estimation_pytorch.models.modules import (
    GatedAttentionUnit,
    ScaleNorm,
)
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator
from deeplabcut.pose_estimation_pytorch.models.weight_init import BaseWeightInitializer


@HEADS.register_module
class RTMCCHead(WeightConversionMixin, BaseHead):
    """RTMPose Coordinate Classification head.

    The RTMCC head is itself adapted from the SimCC head. For more information, see
    "SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation"
    (<https://arxiv.org/pdf/2107.03332>) and "RTMPose: Real-Time Multi-Person Pose
    Estimation based on MMPose" (<https://arxiv.org/pdf/2303.07399>).

    Args:
        input_size: The size of images given to the pose estimation model.
        in_channels: The number of input channels for the head.
        out_channels: Number of channels output by the head (number of bodyparts).
        in_featuremap_size: The size of the input feature map for the head. This is
            equal to the input_size divided by the backbone stride.
        simcc_split_ratio: The split ratio of pixels, as described in SimCC.
        final_layer_kernel_size: Kernel size of the final convolutional layer.
        gau_cfg: Configuration for the GatedAttentionUnit.
        predictor: The predictor for the head. Should usually be a `SimCCPredictor`.
        target_generator: The target generator for the head. Should usually be a
            `SimCCGenerator`.
        criterion: The loss criterions for the RTMCC outputs. There should be a
            criterion for "x" and a criterion for "y".
        aggregator: The loss aggregator to combine the losses.
        weight_init: The weight initializer to use for the head.
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
        criterion: dict[str, BaseCriterion],
        aggregator: BaseLossAggregator,
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
        feats = torch.flatten(feats, start_dim=2)  # -> B, K, hidden=HxW
        feats = self.mlp(feats)  # -> B, K, hidden
        feats = self.gau(feats)
        x, y = self.cls_x(feats), self.cls_y(feats)
        return dict(x=x, y=y)

    @staticmethod
    def convert_weights(
        state_dict: dict[str, torch.Tensor],
        module_prefix: str,
        conversion: torch.Tensor,
        *,
        omit_gau_w: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Re-order / subset bodypart (token) channels for transfer from SuperAnimal.

        Args:
            state_dict: State dict for this head.
            module_prefix: Prefix for state-dict keys.
            conversion: Mapping from new bodyparts to source bodyparts.
            omit_gau_w: If True, remove ``gau.w`` from the returned dict instead of
                constructing a remapped replacement. This requires loading with
                ``strict=False`` to avoid missing-key errors.
                Prefer omitting when source/target keypoint ordering semantics differ.
        """
        conv = conversion.long()
        k_new = int(conv.shape[0])

        # Remap final layer weights and biases if they exist.
        fl_w = f"{module_prefix}final_layer.weight"
        fl_b = f"{module_prefix}final_layer.bias"
        if fl_w in state_dict:
            state_dict[fl_w] = state_dict[fl_w][conv]
        if fl_b in state_dict:
            state_dict[fl_b] = state_dict[fl_b][conv]

        # Remap or re-init gau.w if it exists (only if omit_gau_w is False)
        w_key = f"{module_prefix}gau.w"
        if w_key in state_dict:
            if omit_gau_w:
                state_dict.pop(w_key, None)
                return state_dict

            w_old = state_dict[w_key]
            k_old = (w_old.shape[0] + 1) // 2
            old_center = k_old - 1
            new_center = k_new - 1

            # Deterministic default for unmapped offsets (mean of original weights).
            default_val = w_old.mean()
            w_new = torch.empty(2 * k_new - 1, dtype=w_old.dtype, device=w_old.device)
            for idx_new, d in enumerate(range(-new_center, new_center + 1)):
                old_vals = []
                for i in range(k_new):
                    j = i - d
                    if not (0 <= j < k_new):
                        continue
                    old_idx = int(conv[i] - conv[j]) + old_center
                    if 0 <= old_idx < w_old.shape[0]:
                        old_vals.append(w_old[old_idx])
                w_new[idx_new] = torch.stack(old_vals).mean() if old_vals else default_val
            state_dict[w_key] = w_new
        return state_dict

    @staticmethod
    def update_input_size(model_cfg: dict, input_size: tuple[int, int]) -> None:
        """Updates an RTMPose model configuration file for a new image input size.

        Args:
            model_cfg: The model configuration to update in-place.
            input_size: The updated input (width, height).
        """
        _sigmas = {192: 4.9, 256: 5.66, 288: 6, 384: 6.93}

        def _sigma(size: int) -> float:
            sigma = _sigmas.get(size)
            if sigma is None:
                return 2.87 + 0.01 * size

            return sigma

        w, h = input_size
        model_cfg["data"]["inference"]["top_down_crop"] = dict(width=w, height=h)
        model_cfg["data"]["train"]["top_down_crop"] = dict(width=w, height=h)
        head_cfg = model_cfg["model"]["heads"]["bodypart"]
        head_cfg["input_size"] = input_size
        head_cfg["in_featuremap_size"] = h // 32, w // 32
        head_cfg["target_generator"]["input_size"] = input_size
        head_cfg["target_generator"]["sigma"] = (_sigma(w), _sigma(h))
