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
"""Normalization layers"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    """Implementation of ScaleNorm

    ScaleNorm was introduced in "Transformers without Tears: Improving the Normalization
    of Self-Attention".

    Code based on the `mmpose` implementation. See https://github.com/open-mmlab/mmpose
    for more details.

    Args:
        dim: The dimension of the scale vector.
        eps: The minimum value in clamp.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g
