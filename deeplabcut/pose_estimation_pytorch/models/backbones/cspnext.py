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
from dataclasses import dataclass

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
from deeplabcut.pose_estimation_pytorch.models.modules.csp import (
    CSPConvModule,
    CSPLayer,
    SPPBottleneck,
)


@dataclass(frozen=True)
class CSPNeXtLayerConfig:
    in_channels: int
    out_channels: int
    num_blocks: int
    add_identity: bool
    use_spp: bool


@BACKBONES.register_module
class CSPNeXt(BaseBackbone):
    """TODO: Documentation

    Init weights from AP10k:
    https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#animal-2d-17-keypoints

    Pretrained backbones:
    https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#pretrained-models

    https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/backbones/cspnext.py#L18
    """

    CONFIGS: dict[str, list[CSPNeXtLayerConfig]] = {
        "cspnext_p5": [
            CSPNeXtLayerConfig(64, 128, 3, True, False),
            CSPNeXtLayerConfig(128, 256, 6, True, False),
            CSPNeXtLayerConfig(256, 512, 6, True, False),
            CSPNeXtLayerConfig(512, 1024, 3, False, True),
        ],
        "cspnext_p6": [
            CSPNeXtLayerConfig(64, 128, 3, True, False),
            CSPNeXtLayerConfig(128, 256, 6, True, False),
            CSPNeXtLayerConfig(256, 512, 6, True, False),
            CSPNeXtLayerConfig(512, 768, 3, True, False),
            CSPNeXtLayerConfig(768, 1024, 3, False, True),
        ]
    }

    def __init__(
        self,
        model_name: str = "cspnext_p5",
        pretrained: bool = False,
        expand_ratio: float = 0.5,
        deepen_factor: float = 0.67,
        widen_factor: float = 0.75,
        out_indices: tuple[int, ...] = (4,),
        channel_attention: bool = True,
        norm_layer: str = "SyncBN",
        activation_fn: str = "SiLU",
        **kwargs,
    ) -> None:
        """

        AP10K config
        https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/rtmpose/animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py#L63

        COCO object detection config
        https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py
        """
        super().__init__(stride=32, **kwargs)
        if pretrained:
            raise NotImplementedError()

        if model_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown `CSPNeXT` variant: {model_name}. Must be one of "
                f"{self.CONFIGS.keys()}"
            )

        self.layer_configs = self.CONFIGS[model_name]
        self.stem_out_channels = self.layer_configs[0].in_channels
        self.spp_kernel_sizes = (5, 9, 13)

        self.out_indices = out_indices
        # stem has stride 2
        self.stem = nn.Sequential(
            CSPConvModule(
                in_channels=3,
                out_channels=int(self.stem_out_channels * widen_factor // 2),
                kernel_size=3,
                padding=1,
                stride=2,
                norm_layer=norm_layer,
                activation_fn=activation_fn,
            ),
            CSPConvModule(
                in_channels=int(self.stem_out_channels * widen_factor // 2),
                out_channels=int(self.stem_out_channels * widen_factor // 2),
                kernel_size=3,
                padding=1,
                stride=1,
                norm_layer=norm_layer,
                activation_fn=activation_fn,
            ),
            CSPConvModule(
                in_channels=int(self.stem_out_channels * widen_factor // 2),
                out_channels=int(self.stem_out_channels * widen_factor),
                kernel_size=3,
                padding=1,
                stride=1,
                norm_layer=norm_layer,
                activation_fn=activation_fn,
            )
        )
        self.layers = ["stem"]

        for i, layer_cfg in enumerate(self.layer_configs):
            layer_cfg: CSPNeXtLayerConfig
            in_channels = int(layer_cfg.in_channels * widen_factor)
            out_channels = int(layer_cfg.out_channels * widen_factor)
            num_blocks = max(round(layer_cfg.num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = CSPConvModule(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                activation_fn=activation_fn,
            )
            stage.append(conv_layer)
            if layer_cfg.use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=self.spp_kernel_sizes,
                    norm_layer=norm_layer,
                    activation_fn=activation_fn,
                )
                stage.append(spp)

            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=layer_cfg.add_identity,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                norm_layer=norm_layer,
                activation_fn=activation_fn,
            )
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Original code: x: tuple[tensor] -> torch.Tensor

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

        """
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        # TODO(niels): choose which layer we output
        return outs[-1]
