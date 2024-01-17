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
from __future__ import annotations

import torch


def count_nonzero_elems(
    losses: torch.Tensor, weights: float | torch.Tensor, per_batch: bool = False
):
    """
    Compute the number of elements in the loss function induced by `weights`.
    This is a torch implementation of https://github.com/tensorflow/tensorflow/blob/4dacf3f368eb7965e9b5c3bbdd5193986081c3b2/tensorflow/python/ops/losses/losses_impl.py#L89

    Args:
        losses (Tensor): Tensor of shape [batch_size, d1, ... dN].
        weights (Tensor): Tensor of shape [], [batch_size] or [batch_size, d1, ... dK], where K < N.
        per_batch (bool): Whether to return the number of elements per batch or as a sum total.

    Returns:
        Tensor: The number of present (non-zero) elements in the losses tensor.
    """
    if isinstance(weights, float):
        if weights != 0.0:
            return losses.numel()
        else:
            return torch.tensor(0)

    weights = torch.as_tensor(weights, dtype=torch.float32)

    # Check for non-zero weights and broadcast to match losses
    present = torch.where(
        weights == 0.0, torch.zeros_like(weights), torch.ones_like(weights)
    )
    present = present.expand_as(losses)

    # Reduce sum across the desired dimensions
    if per_batch:
        reduction_dims = tuple(range(1, present.dim()))
        return torch.sum(present, dim=reduction_dims, keepdim=True)
    else:
        return torch.sum(present)
