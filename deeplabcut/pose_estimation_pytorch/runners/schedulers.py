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
from torch.optim.lr_scheduler import _LRScheduler


class LRListScheduler(_LRScheduler):
    """
    Definition of the class object Scheduler.
    You can achieve increased performance and faster training by using a learning rate that changes
    during training. A scheduler makes the learning rate adaptative. Given a list of learning rates
    and milestones modifies the learning rate accordingly during training
    """

    def __init__(self, optimizer, milestones, lr_list, last_epoch=-1) -> None:
        """
        Args:
            optimizer: optimizer used for learning.
            milestones: number of epochs.
            lr_list: learning rate list.
            last_epoch: where to start the scheduler. (-1: start from beginning)

        Examples:
            input:
                last_epoch = -1
                verbose = False
                milestones = [10, 30, 40]
                lr_list = [[0.00001],[0.000005],[0.000001]]
        """
        self.milestones = milestones
        self.lr_list = lr_list
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Summary:
        Given a milestones, get the corresponding learning rate.

        Returns:
            lr: learning rate value

        Examples:
            input: LRListScheduler object
            output: learning rate (lr) = [0.001]
        """
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [lr for lr in self.lr_list[self.milestones.index(self.last_epoch)]]


def build_scheduler(
    scheduler_cfg: dict | None, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Builds a scheduler from a configuration, if defined

    Args:
        scheduler_cfg: the configuration of the scheduler to build
        optimizer: the optimizer the scheduler will be built for

    Returns:
        None if scheduler_cfg is None, otherwise the scheduler
    """
    if scheduler_cfg is None:
        return None

    if scheduler_cfg["type"] == "LRListScheduler":
        scheduler = LRListScheduler
    else:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg["type"])

    return scheduler(optimizer=optimizer, **scheduler_cfg["params"])
