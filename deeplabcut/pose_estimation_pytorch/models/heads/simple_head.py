import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.heads.base import HEADS
from .base import BaseHead


@HEADS.register_module
class SimpleHead(BaseHead):
    """
        Deconvolutional head to predict maps from the extracted features
    """

    def __init__(self, channels: list,
                 kernel_size: list,
                 strides: list,
                 pretrained: str = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides

        if len(kernel_size) == 1:
            self.model = self._make_layer(channels[0],
                                          channels[1],
                                          kernel_size[0],
                                          strides[0])
        else:
            layers = []
            for i in range(len(channels) - 1):
                up_layer = self._make_layer(channels[i],
                                            channels[i + 1],
                                            kernel_size[i],
                                            strides[i]
                                            )
                layers.append(up_layer)
                if i < len(channels) - 2:
                    layers.append(nn.ReLU())
            self.model = nn.Sequential(*layers)

        self._init_weights(pretrained)

    def _make_layer(self,
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride):
        upsample_layer = nn.ConvTranspose2d(input_channels, output_channels,
                                            kernel_size, stride=stride)
        return upsample_layer

    def forward(self, x):
        out = self.model(x)

        return out

# class TransformerHead(BaseHead):
#
#     def __init__(self, dim, hidden_heatmap_dim,
#                  heatmap_dim, apply_multi,
#                  heatmap_size,
#                  apply_init):
#         super().__init__()
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim * 3),
#             nn.Linear(dim * 3, hidden_heatmap_dim),
#             nn.LayerNorm(hidden_heatmap_dim),
#                         nn.Linear(hidden_heatmap_dim, heatmap_dim)
#         ) if (dim*3 <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
#             nn.LayerNorm(dim*3),
#             nn.Linear(dim*3, heatmap_dim)
#         )
#         self.heatmap_size = heatmap_size
#         # trunc_normal_(self.keypoint_token, std=.02)
#         if apply_init:
#             self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x):
#         x = self.mlp_head(x)
#         x = rearrange(x,'b c (p1 p2) -> b c p1 p2',
#                       p1=self.heatmap_size[0], p2=self.heatmap_size[1])
#
#         return x
