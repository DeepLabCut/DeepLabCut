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
from typing import Tuple

import torch
from einops import rearrange, repeat
from timm.layers import trunc_normal_

from deeplabcut.pose_estimation_pytorch.models.necks.base import BaseNeck, NECKS
from deeplabcut.pose_estimation_pytorch.models.necks.layers import TransformerLayer
from deeplabcut.pose_estimation_pytorch.models.necks.utils import (
    make_sine_position_embedding,
)

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


@NECKS.register_module
class Transformer(BaseNeck):
    """Transformer Neck for pose estimation.
       title={TokenPose: Learning Keypoint Tokens for Human Pose Estimation},
       author={Yanjie Li and Shoukui Zhang and Zhicheng Wang and Sen Yang and Wankou Yang and Shu-Tao Xia and Erjin Zhou},
       booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
       year={2021}

    Args:
        feature_size: Size of the input feature map (height, width).
        patch_size: Size of each patch used in the transformer.
        num_keypoints: Number of keypoints in the pose estimation task.
        dim: Dimension of the transformer.
        depth: Number of transformer layers.
        heads: Number of self-attention heads in the transformer.
        mlp_dim: Dimension of the MLP used in the transformer.
                                 Defaults to 3.
        apply_init: Whether to apply weight initialization.
                                     Defaults to False.
        heatmap_size: Size of the heatmap. Defaults to [64, 64].
        channels: Number of channels in each patch. Defaults to 32.
        dropout: Dropout rate for embeddings. Defaults to 0.0.
        emb_dropout: Dropout rate for transformer layers.
                                       Defaults to 0.0.
        pos_embedding_type: Type of positional embedding.
                            Either 'sine-full', 'sine', or 'learnable'.
                            Defaults to "sine-full".

    Examples:
        # Creating a Transformer neck with sine positional embedding
        transformer = Transformer(
            feature_size=(128, 128),
            patch_size=(16, 16),
            num_keypoints=17,
            dim=256,
            depth=6,
            heads=8,
            pos_embedding_type="sine"
        )

        # Creating a Transformer neck with learnable positional embedding
        transformer = Transformer(
            feature_size=(256, 256),
            patch_size=(32, 32),
            num_keypoints=17,
            dim=512,
            depth=12,
            heads=16,
            pos_embedding_type="learnable"
        )
    """

    def __init__(
        self,
        *,
        feature_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_keypoints: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int = 3,
        apply_init: bool = False,
        heatmap_size: Tuple[int, int] = (64, 64),
        channels: int = 32,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        pos_embedding_type: str = "sine-full"
    ):
        super().__init__()

        num_patches = (feature_size[0] // (patch_size[0])) * (
            feature_size[1] // (patch_size[1])
        )
        patch_dim = channels * patch_size[0] * patch_size[1]

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = self.pos_embedding_type == "sine-full"

        self.keypoint_token = torch.nn.Parameter(
            torch.zeros(1, self.num_keypoints, dim)
        )
        h, w = (
            feature_size[0] // (self.patch_size[0]),
            feature_size[1] // (self.patch_size[1]),
        )

        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = torch.nn.Linear(patch_dim, dim)
        self.dropout = torch.nn.Dropout(emb_dropout)

        self.transformer1 = TransformerLayer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            scale_with_head=True,
        )
        self.transformer2 = TransformerLayer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
        )
        self.transformer3 = TransformerLayer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
        )

        self.to_keypoint_token = torch.nn.Identity()

        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(
        self, w: int, h: int, d_model: int, pe_type="learnable"
    ):
        """Create position embeddings for the transformer.

        Args:
            w: Width of the input feature map.
            h: Height of the input feature map.
            d_model: Dimension of the transformer encoder.
            pe_type: Type of position embeddings.
                     Either "learnable" or "sine". Defaults to "learnable".
        """
        with torch.no_grad():
            self.pe_h = h
            self.pe_w = w
            length = h * w
        if pe_type != "learnable":
            self.pos_embedding = torch.nn.Parameter(
                make_sine_position_embedding(h, w, d_model), requires_grad=False
            )
        else:
            self.pos_embedding = torch.nn.Parameter(
                torch.zeros(1, self.num_patches + self.num_keypoints, d_model)
            )

    def _make_layer(
        self, block: torch.nn.Module, planes: int, blocks: int, stride: int = 1
    ) -> torch.nn.Sequential:
        """Create a layer of the transformer encoder.

        Args:
            block: The basic building block of the layer.
            planes: Number of planes in the layer.
            blocks: Number of blocks in the layer.
            stride: Stride value. Defaults to 1.

        Returns:
            The layer of the transformer encoder.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def _init_weights(self, m: torch.nn.Module):
        """Initialize the weights of the model.

        Args:
            m: A module of the model.
        """
        print("Initialization...")
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, feature: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass through the Transformer neck.

        Args:
            feature: Input feature map.
            mask: Mask to apply to the transformer.
                  Defaults to None.

        Returns:
            Output tensor from the transformer neck.

        Examples:
            # Assuming feature is a torch.Tensor of shape (batch_size, channels, height, width)
            output = transformer(feature)
        """
        p = self.patch_size

        x = rearrange(
            feature, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p[0], p2=p[1]
        )
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, "() n d -> b n d", b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + self.num_keypoints)]
        x = self.dropout(x)

        x1 = self.transformer1(x, mask, self.pos_embedding)
        x2 = self.transformer2(x1, mask, self.pos_embedding)
        x3 = self.transformer3(x2, mask, self.pos_embedding)

        x1_out = self.to_keypoint_token(x1[:, 0 : self.num_keypoints])
        x2_out = self.to_keypoint_token(x2[:, 0 : self.num_keypoints])
        x3_out = self.to_keypoint_token(x3[:, 0 : self.num_keypoints])

        x = torch.cat((x1_out, x2_out, x3_out), dim=2)
        return x
