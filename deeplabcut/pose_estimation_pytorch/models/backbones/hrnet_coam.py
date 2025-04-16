# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import BACKBONES
from deeplabcut.pose_estimation_pytorch.models.backbones.hrnet import HRNet
from deeplabcut.pose_estimation_pytorch.models.modules import (  # ColoredKeypointEncoder,; StackedKeypointEncoder,
    BaseKeypointEncoder,
    CoAMBlock,
    KEYPOINT_ENCODERS,
    SelfAttentionModule_CoAM,
)


@BACKBONES.register_module
class HRNetCoAM(HRNet):
    """HRNet backbone with Conditional Attention Module (CoAM).

    This version returns high-resolution feature maps of size 1/4 * original_image_size.

    Attributes:
        model: the HRNet model
        coam_stages: CoAM blocks for each stage
    """

    def __init__(
        self,
        kpt_encoder: dict | BaseKeypointEncoder,
        base_model_name: str = "hrnet_w32",
        pretrained: bool = True,
        coam_modules: tuple[int, ...] = (2,),
        selfatt_coam_modules: tuple[int, ...] | None = None,
        channel_att_only: bool = False,
        att_heads: int = 1,
        img_size: tuple[int, int] = (256, 256),
        **kwargs,
    ) -> None:
        """Constructs an ImageNet pretrained HRNet from timm and creates CoAM blocks.

        Args:
            base_model_name: Type of HRNet (e.g., 'hrnet_w32', 'hrnet_w48').
            pretrained: If True, loads the model with ImageNet pretrained weights.
            coam_modules: List of stages to apply CoAM.
            selfatt_coam_modules: List of stages to apply Self-Attention-CoAM.
            channel_att_only: Whether to use only channel attention block in CoAM.
            att_heads: Number of attention heads.
            cond_enc: Type of conditional encoding ('stacked', 'colored', or greyscale).
            img_size: The (height, width) size of the input images.
            num_joints: Number of joints in the dataset.
        """

        super().__init__(model_name=base_model_name, pretrained=pretrained, **kwargs)

        self.coam_modules = coam_modules
        self.selfatt_coam_modules = selfatt_coam_modules
        self.channel_att_only = channel_att_only
        if not isinstance(kpt_encoder, BaseKeypointEncoder):
            if "img_size" not in kpt_encoder:
                kpt_encoder["img_size"] = img_size
            kpt_encoder = KEYPOINT_ENCODERS.build(kpt_encoder)

        self.cond_enc = kpt_encoder

        self.coam_stages = nn.ModuleList([None, None, None, None])
        self.selfatt_coam_stages = nn.ModuleList([None, None, None, None])

        spat_dims = [
            (int(img_size[0] / 4), int(img_size[1] / 4)),
            (int(img_size[0] / 8), int(img_size[1] / 8)),
            (int(img_size[0] / 16), int(img_size[1] / 16)),
            (int(img_size[0] / 32), int(img_size[1] / 32)),
        ]

        assert not (
            set(coam_modules) & set(selfatt_coam_modules)
            if selfatt_coam_modules
            else set()
        ), "CoAM and Self-Attention-CoAM cannot be used at the same time"

        all_output_channels = [
            self.model.stage2_cfg["num_channels"],
            self.model.stage3_cfg["num_channels"],
            self.model.stage4_cfg["num_channels"],
        ]

        for coam_pos in self.coam_modules:
            if coam_pos == 4:
                spat_dims_ = [spat_dims[0]]
                channels = [all_output_channels[-1][0]]
            else:
                spat_dims_ = spat_dims[: coam_pos + 1]
                channels = all_output_channels[coam_pos - 1]

            self.coam_stages[coam_pos - 1] = CoAMBlock(
                spat_dims=spat_dims_,
                channel_list=channels,
                cond_enc=self.cond_enc,
                n_heads=att_heads,
                channel_only=self.channel_att_only,
            )

        if self.selfatt_coam_modules:
            for selfatt_coam_pos in self.selfatt_coam_modules:
                if selfatt_coam_pos == 4:
                    spat_dims_ = [spat_dims[0]]
                    channels = [all_output_channels[-1][0]]
                else:
                    spat_dims_ = spat_dims[: selfatt_coam_pos + 1]
                    channels = all_output_channels[coam_pos - 1]
                self.selfatt_coam_stages[selfatt_coam_pos - 1] = (
                    SelfAttentionModule_CoAM(
                        spat_dims=spat_dims_, channel_list=channels
                    )
                )

    def stages(self, x, cond_hm) -> list[torch.Tensor]:
        x = self.model.layer1(x)

        xl = [t(x) for i, t in enumerate(self.model.transition1)]

        if self.coam_stages[0]:
            xl = self.coam_stages[0](xl, cond_hm)
        elif self.selfatt_coam_stages[0]:
            xl = self.selfatt_coam_stages[0](xl)

        yl = self.model.stage2(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.model.transition2)
        ]

        if self.coam_stages[1]:
            xl = self.coam_stages[1](xl, cond_hm)
        elif self.selfatt_coam_stages[1]:
            xl = self.selfatt_coam_stages[1](xl)

        yl = self.model.stage3(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.model.transition3)
        ]

        if self.coam_stages[2]:
            xl = self.coam_stages[2](xl, cond_hm)
        elif self.selfatt_coam_stages[2]:
            xl = self.selfatt_coam_stages[2](xl)

        yl = self.model.stage4(xl)

        if self.coam_stages[3]:
            yl = self.coam_stages[3](yl, cond_hm)
        elif self.selfatt_coam_stages[3]:
            yl = self.selfatt_coam_stages[3](yl)

        return yl

    def forward(self, x: torch.Tensor, cond_kpts: np.ndarray):
        """Forward pass through the HRNetCoAM backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            cond_kpts: Conditional keypoints of shape (batch_size, num_joints, 2).

        Returns:
            the feature map
        """

        # create conditional heatmap
        if isinstance(cond_kpts, torch.Tensor):
            cond_kpts = cond_kpts.detach().numpy()
        cond_hm = self.cond_enc(cond_kpts.squeeze(1), x.size()[2:])
        cond_hm = torch.from_numpy(cond_hm).float().to(x.device)
        cond_hm = cond_hm.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Stem
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.act2(x)

        # Stages
        y = self.stages(x, cond_hm)

        if self.model.incre_modules is not None:
            raise NotImplementedError(
                "Incremental HRNet modules not supported for HRNetCoAM"
            )
            x = [incre(f) for f, incre in zip(x, self.model.incre_modules)]

        return self.prepare_output(y)
