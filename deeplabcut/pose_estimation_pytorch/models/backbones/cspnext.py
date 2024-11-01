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
"""Implementation of the CSPNeXt Backbone

Based on the ``mmdetection`` CSPNeXt implementation. For more information, see:
<https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/cspnext.py>

For more details about this architecture, see `RTMDet: An Empirical Study of Designing
Real-Time Object Detectors`: https://arxiv.org/abs/1711.05101.
"""
from dataclasses import dataclass
from pathlib import Path

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
    """Configuration for a CSPNeXt layer"""
    in_channels: int
    out_channels: int
    num_blocks: int
    add_identity: bool
    use_spp: bool


@BACKBONES.register_module
class CSPNeXt(BaseBackbone):
    """CSPNeXt Backbone

    Args:
        model_name: The model variant to build. Must be one of the keys of the
            ``CSPNeXt.CONFIGS`` attribute (e.g. `cspnext_p5`, `cspnext_p6`, ...).
        pretrained: Whether to load pre-trained weights for the model.
        expand_ratio: Ratio used to adjust the number of channels of the hidden layer.
        deepen_factor: Number of blocks in each CSP layer is multiplied by this value.
        widen_factor: Number of channels in each layer is multiplied by this value.
        out_indices: The branch indices to output. If a tuple of integers, the outputs
            are returned as a list of tensors. If a single integer, a tensor is returned
            containing the configured index.
        channel_attention: Add chanel attention to all stages
        norm_layer: The type of normalization layer to use.
        activation_fn: The type of activation function to use.
        pretrained_weights: TODO(niels) load the pretrained weights automatically
        **kwargs: BaseBackbone kwargs.
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
        out_indices: int | tuple[int, ...] = -1,
        channel_attention: bool = True,
        norm_layer: str = "SyncBN",
        activation_fn: str = "SiLU",
        pretrained_weights: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(stride=32, **kwargs)
        if model_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown `CSPNeXT` variant: {model_name}. Must be one of "
                f"{self.CONFIGS.keys()}"
            )

        self.layer_configs = self.CONFIGS[model_name]
        self.stem_out_channels = self.layer_configs[0].in_channels
        self.spp_kernel_sizes = (5, 9, 13)

        self.single_output = isinstance(out_indices, int)
        if self.single_output:
            out_indices = (out_indices,)

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

        if pretrained:
            # FIXME(niels): upload CSPNext weights somewhere and download them
            if pretrained_weights is None:
                raise ValueError("When pretrained=True, pretrained_weights must be set")

            self._from_pretrained(pretrained_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if self.single_output:
            return outs[-1]

        return tuple(outs)

    def _from_pretrained(self, weight_path: str | Path) -> None:
        """FIXME(niels): download weights from somewhere"""
        snapshot = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.load_state_dict(snapshot["state_dict"])
