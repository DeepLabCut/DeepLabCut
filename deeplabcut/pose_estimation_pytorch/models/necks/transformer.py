import torch
from einops import rearrange, repeat
from torch import nn
from timm.layers import trunc_normal_
from .layers import TransformerLayer
from .utils import make_sine_position_embedding
from .base import NECKS

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


@NECKS.register_module
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        feature_size,
        patch_size,
        num_keypoints,
        dim,
        depth,
        heads,
        mlp_dim=3,
        apply_init=False,
        heatmap_size=[64, 64],
        channels=32,
        dropout=0.0,
        emb_dropout=0.0,
        pos_embedding_type="sine-full"
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

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (
            self.patch_size[1]
        )

        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

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

        self.to_keypoint_token = nn.Identity()

        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type="learnable"):
        """
        d_model: embedding size in transformer encoder
        """
        with torch.no_grad():
            self.pe_h = h
            self.pe_w = w
            length = h * w
        if pe_type != "learnable":
            self.pos_embedding = nn.Parameter(
                make_sine_position_embedding(h, w, d_model), requires_grad=False
            )
        else:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.num_patches + self.num_keypoints, d_model)
            )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask=None):
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
