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

from typing import Any

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LRListScheduler(_LRScheduler):
    """
    You can achieve increased performance and faster training by using a learning rate
    that changes during training. A scheduler makes the learning rate adaptive. Given a
    list of learning rates and milestones modifies the learning rate accordingly during
    training.
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

    parsed_params = {}
    for param_name, param in scheduler_cfg["params"].items():
        if isinstance(param, list):
            param = [_parse_scheduler_param(p, optimizer) for p in param]
        else:
            param = _parse_scheduler_param(param, optimizer)

        parsed_params[param_name] = param

    return scheduler(optimizer=optimizer, **parsed_params)


def _parse_scheduler_param(param: Any, optimizer: torch.optim.Optimizer) -> Any:
    """Parses parameters so they're built as schedulers if they're configured as one"""
    if isinstance(param, dict) and "type" in param:
        param = build_scheduler(param, optimizer)

    return param


def load_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state_dict: dict,
) -> None:
    """
    Args:
        scheduler: The scheduler for which to load the state dict.
        state_dict: The state dict to load

    Raises:
        ValueError: if the state dict fails to load.
    """
    try:
        scheduler.load_state_dict(state_dict)
    except Exception as err:
        raise ValueError(f"Failed to load state dict: {err}")

    param_groups = scheduler.optimizer.param_groups
    resume_lrs = scheduler.get_last_lr()

    if len(param_groups) != len(resume_lrs):
        raise ValueError(
            f"Number of optimizer parameter groups ({len(param_groups)}) did not match "
            f"number of learning rates to resume from ({len(scheduler.get_last_lr())})."
        )

    # Update the learning rate for the optimizer based on the scheduler
    for group, resume_lr in zip(param_groups, resume_lrs):
        group['lr'] = resume_lr
