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
"""Gated Attention Unit

Based on the building blocks used for the ``mmdetection`` CSPNeXt implementation. For
more information, see <https://github.com/open-mmlab/mmdetection>.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.layers as timm_layers

from deeplabcut.pose_estimation_pytorch.models.modules.norm import ScaleNorm


def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i

    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device), spatial_shape
    )

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(
        half_size
    )
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Scale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim: The dimension of the scale vector.
        init_value: The initial value of the scale vector.
        trainable: Whether the scale vector is trainable.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock"""

    def __init__(
        self,
        num_token,
        in_token_dims,
        out_token_dims,
        expansion_factor=2,
        s=128,
        eps=1e-5,
        dropout_rate=0.0,
        drop_path=0.0,
        attn_type="self-attn",
        act_fn="SiLU",
        bias=False,
        use_rel_bias=True,
        pos_enc=False,
    ):
        super(GatedAttentionUnit, self).__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc

        if drop_path > 0.0:
            self.drop_path = timm_layers.DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == "self-attn":
                self.w = nn.Parameter(
                    torch.rand([2 * num_token - 1], dtype=torch.float)
                )
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == "self-attn":
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == "SiLU" or act_fn == nn.SiLU:
            self.act_fn = nn.SiLU(True)
        elif act_fn == "ReLU" or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""

        if self.attn_type == "self-attn":
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function."""

        if self.attn_type == "self-attn":
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == "self-attn":
            # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            if self.pos_enc:
                base = rope(base, dim=1)
            # [B, K, 2, s] -> [B, K, s], [B, K, s]
            q, k = torch.unbind(base, dim=2)

        else:
            # [B, K, e + s] -> [B, K, e], [B, K, s]
            u, q = torch.split(uv, [self.e, self.s], dim=2)

            k = self.k_fc(k)  # -> [B, K, s]
            v = self.v_fc(v)  # -> [B, K, e]

            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # [B, K, s].permute() -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            if self.attn_type == "self-attn":
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, : q.size(1), : k.size(1)]
        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)
        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.bmm(kernel, v)

        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x):
        if self.shortcut:
            if self.attn_type == "cross-attn":
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))
