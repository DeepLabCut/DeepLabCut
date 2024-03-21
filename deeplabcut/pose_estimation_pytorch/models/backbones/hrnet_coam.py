# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
)
from deeplabcut.pose_estimation_pytorch.models.backbones.hrnet import (
    HRNet,
)
from deeplabcut.pose_estimation_pytorch.models.modules import (
    CoAMBlock,
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
            base_model_name: str = "hrnet_w32",
            pretrained: bool = True,
            coam_modules: tuple[int,...] = (2,),
            selfatt_coam_modules: tuple[int,...] | None = None,
            channel_att_only: bool = False,
            att_heads: int = 1,
            cond_enc: str = 'colored',
            img_size: tuple[int,int] = (256,  256),
            num_joints: int = 17,
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
            img_size: Size of the input image.
            num_joints: Number of joints in the dataset.
        """

        super().__init__(
            model_name = base_model_name,
            pretrained = pretrained,
            only_high_res = True,
            **kwargs)

        self.coam_modules = coam_modules
        self.selfatt_coam_modules = selfatt_coam_modules
        self.channel_att_only = channel_att_only
        self.cond_enc = cond_enc

        self.coam_stages = [None, None, None, None]
        self.selfatt_coam_stages = [None, None, None, None]

        spat_dims = [(int(img_size[0]/4),int(img_size[1]/4)),
                     (int(img_size[0]/8),int(img_size[1]/8)),
                     (int(img_size[0]/16),int(img_size[1]/16)),
                     (int(img_size[0]/32),int(img_size[1]/32))]

        assert not(set(coam_modules) & set(selfatt_coam_modules) if selfatt_coam_modules else set()), \
            "CoAM and Self-Attention-CoAM cannot be used at the same time"

        all_output_channels = [self.model.stage2_cfg['num_channels'],
                               self.model.stage3_cfg['num_channels'],
                               self.model.stage4_cfg['num_channels']]

        for coam_pos in self.coam_modules:
            if coam_pos == 4:
                spat_dims_ = [spat_dims[0]]
                channels = [all_output_channels[-1][0]]
            else:
                spat_dims_ = spat_dims[:coam_pos+1]
                channels = all_output_channels[coam_pos-1]
            
            self.coam_stages[coam_pos-1] = CoAMBlock(spat_dims=spat_dims_, channel_list=channels,
                                                     cond_stacked=self.cond_enc, num_joints = num_joints,
                                                     n_heads=att_heads, channel_only=self.channel_att_only)
        
        if self.selfatt_coam_modules:
            for selfatt_coam_pos in self.selfatt_coam_modules:
                if selfatt_coam_pos == 4:
                    spat_dims_ = [spat_dims[0]]
                    channels = [all_output_channels[-1][0]]
                else:
                    spat_dims_ = spat_dims[:selfatt_coam_pos+1]
                    channels = all_output_channels[coam_pos-1]
                self.selfatt_coam_stages[selfatt_coam_pos-1] = SelfAttentionModule_CoAM(spat_dims=spat_dims_, channel_list=channels)


    def stages(self, x, cond_hm) -> list[torch.Tensor]:
        x = self.model.layer1(x)

        xl = [t(x) for i, t in enumerate(self.model.transition1)]

        if self.coam_stages[0]:
            xl = self.coam_stages[0](xl, cond_hm)
        elif self.selfatt_coam_modules[0]:
            xl = self.selfatt_coam_stages[0](xl)

        yl = self.model.stage2(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.model.transition2)]
        
        if self.coam_stages[1]:
            xl = self.coam_stages[1](xl, cond_hm)
        elif self.selfatt_coam_modules[1]:
            xl = self.selfatt_coam_stages[1](xl)
        
        yl = self.model.stage3(xl)

        xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.model.transition3)]

        if self.coam_stages[2]:
            xl = self.coam_stages[2](xl, cond_hm)
        elif self.selfatt_coam_modules[2]:
            xl = self.selfatt_coam_stages[2](xl)

        yl = self.model.stage4(xl)

        if self.coam_stages[3]:
            yl = self.coam_stages[3](yl, cond_hm)
        elif self.selfatt_coam_modules[3]:
            yl = self.selfatt_coam_stages[3](yl)

        return yl
    

    def forward(self, x):
        """Forward pass through the HRNetCoAM backbone.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width, condition_channels).

        Returns:
            the feature map

        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import HRNetCoAM
            >>> backbone = HRNetCoAM(model_name='hrnet_w32', pretrained=False)
            >>> x = torch.randn(1, 6, 256, 256)
            >>> y = backbone(x)
        """
        
        if x[:,3:].shape[1] == 0:
            raise Exception("condition is empty, please check your dataloader")
        x_ = x[:,:3]
        cond_hm = x[:,3:]
        # TODO: cond_hm = self.cond_encoder(cond_hm)

        # Stem
        x = self.model.conv1(x_)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.act2(x)

        # Stages
        y = self.stages(x, cond_hm)

        if self.model.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.model.incre_modules)]

        return self.prepare_output(y)
