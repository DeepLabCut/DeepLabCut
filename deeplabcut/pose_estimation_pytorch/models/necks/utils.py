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

import math

import torch


def make_sine_position_embedding(
    h: int, w: int, d_model: int, temperature: int = 10000, scale: float = 2 * math.pi
) -> torch.Tensor:
    """Generate sine position embeddings for a given height, width, and model dimension.

    Args:
        h: Height of the embedding.
        w: Width of the embedding.
        d_model: Dimension of the model.
        temperature: Temperature parameter for position embedding calculation.
                     Defaults to 10000.
        scale: Scaling factor for position embedding. Defaults to 2 * math.pi.

    Returns:
        Sine position embeddings with shape (batch_size, d_model, h * w).

    Example:
        >>> h, w, d_model = 10, 20, 512
        >>> pos_emb = make_sine_position_embedding(h, w, d_model)
        >>> print(pos_emb.shape)  # Output: torch.Size([1, 512, 200])
    """
    area = torch.ones(1, h, w)
    y_embed = area.cumsum(1, dtype=torch.float32)
    x_embed = area.cumsum(2, dtype=torch.float32)
    one_direction_feats = d_model // 2
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    pos = pos.flatten(2).permute(0, 2, 1)

    return pos
