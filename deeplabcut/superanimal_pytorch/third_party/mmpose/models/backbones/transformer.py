from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as ViT
from mmpose.utils import get_root_logger

from .base_transformer import TokenPose_S_base, TokenPose_TB_base
from .utils import load_checkpoint
from ..builder import BACKBONES

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class TokenPose_S(nn.Module):
    def __init__(
        self,
        IMAGE_SIZE,
        PATCH_SIZE,
        NUM_JOINTS,
        DIM,
        TRANSFORMER_DEPTH,
        TRANSFORMER_HEADS,
        TRANSFORMER_MLP_RATIO,
        HEATMAP_SIZE,
        POS_EMBEDDING_TYPE,
    ):

        super(TokenPose_S, self).__init__()

        self.features = TokenPose_S_base(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_keypoints=NUM_JOINTS,
            channels=256,
            depth=TRANSFORMER_DEPTH,
            heads=TRANSFORMER_HEADS,
            mlp_dim=DIM * TRANSFORMER_MLP_RATIO,
            dim=DIM,
            hidden_heatmap_dim=HEATMAP_SIZE[1] * HEATMAP_SIZE[0] // 8,
            heatmap_dim=HEATMAP_SIZE[1] * HEATMAP_SIZE[0],
            heatmap_size=[HEATMAP_SIZE[1], HEATMAP_SIZE[0]],
            pos_embedding_type=POS_EMBEDDING_TYPE,
        )

    def forward(self, x):
        x = self.features(x)
        return x

    def init_weights(self, pretrained=""):
        pass


@BACKBONES.register_module()
class TokenPose_T(nn.Module):
    def __init__(
        self,
        IMAGE_SIZE,
        PATCH_SIZE,
        NUM_JOINTS,
        DIM,
        BASE_CHANNEL,
        TRANSFORMER_DEPTH,
        TRANSFORMER_HEADS,
        TRANSFORMER_MLP_RATIO,
        HIDDEN_HEATMAP_DIM,
        HEATMAP_SIZE,
        POS_EMBEDDING_TYPE,
    ):

        super(TokenPose_T, self).__init__()

        self.transformer = TokenPose_TB_base(
            feature_size=[IMAGE_SIZE[1], IMAGE_SIZE[0]],
            patch_size=[PATCH_SIZE[1], PATCH_SIZE[0]],
            num_keypoints=NUM_JOINTS,
            dim=DIM,
            channels=BASE_CHANNEL,
            depth=TRANSFORMER_DEPTH,
            heads=TRANSFORMER_HEADS,
            mlp_dim=DIM * TRANSFORMER_MLP_RATIO,
            hidden_heatmap_dim=HIDDEN_HEATMAP_DIM,
            heatmap_dim=HEATMAP_SIZE[1] * HEATMAP_SIZE[0],
            heatmap_size=[HEATMAP_SIZE[1], HEATMAP_SIZE[0]],
            pos_embedding_type=POS_EMBEDDING_TYPE,
        )

    def forward(self, x):
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained=""):
        pass


def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_S(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


@BACKBONES.register_module()
class VisionTransformer(ViT):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_fpn=False,
    ):

        super(VisionTransformer, self).__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )
        self.use_fpn = use_fpn
        self.patch_size = patch_size
        if use_fpn:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(
                    embed_dim, embed_dim, kernel_size=6, stride=2, padding=0
                ),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        Hp = int(((H * W) / self.patch_size ** 2) ** 0.5)
        Wp = int(((H * W) / self.patch_size ** 2) ** 0.5)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        outcome = x[:, 1:]
        if self.use_fpn:
            outcome = outcome.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            features = []
            for i in range(len(ops)):
                features.append(ops[i](outcome))
            return tuple(features)
        else:
            return outcome

    def init_weights(self, pretrained=None):
        """
        Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None and pretrained != "":
            load_checkpoint(
                self,
                pretrained,
            )
        # for l in self.blocks:
        #     for name, p in l.named_children():
        #         if "norm" not in name:
        #             for part_name, r in p.named_children():
        #                 print(part_name)
        #                 try:
        #                     r.weight.requires_grad = False
        #                 except Exception:
        #                     pass
        # else:
        #     pass
