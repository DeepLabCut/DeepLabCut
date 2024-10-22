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
"""TODO: Source"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplabcut.pose_estimation_pytorch.models.criterions.base import (
    BaseCriterion,
    CRITERIONS,
)


@CRITERIONS.register_module
class KLDiscreteLoss(BaseCriterion):
    """
    TODO: https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/mmpose/models/losses/classification_loss.py#L142
    """

    def __init__(
        self,
        beta=1.0,
        label_softmax=False,
        label_beta=10.0,
        use_target_weight=True,
        mask=None,
        mask_weight=1.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="none")

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        n, k, _ = output.shape
        if self.use_target_weight and isinstance(weights, torch.Tensor):
            weight = weights.reshape(-1)
        else:
            weight = 1.0

        pred = output.reshape(-1, output.size(-1))
        target = target.reshape(-1, target.size(-1))
        loss = self.criterion(pred, target).mul(weight)
        if self.mask is not None:
            loss = loss.reshape(n, k)
            loss[:, self.mask] = loss[:, self.mask] * self.mask_weight

        return loss.sum() / k

    def criterion(self, dec_outs, labels):
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        return loss
