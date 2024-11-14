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

import random
from dataclasses import dataclass

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners.schedulers as schedulers


def generate_random_lr_list(num_floats: int):
    """Summary:
    Generate list of lists including random numbers.

    Args:
        num_floats: number of floats we want to include in our list

    Returns:
        ran_list: random list of sorted numbers, being first number bigger than the last

    Examples:
        input: num_float = 2
        output: [[0.96420871896179], [0.3917365732012833]]
    """
    ran_list = []
    for i in range(num_floats):
        random_floats = [random.random()]
        ran_list.append(random_floats)
    return sorted(ran_list, reverse=True)


@pytest.mark.parametrize(
    "milestones, lr_list",
    [
        ([10, 430], [[0.05], [0.005]]),
        (list(sorted(random.sample(range(0, 999), 2))), generate_random_lr_list(2))
    ]
)
def test_scheduler(milestones, lr_list):
    """Summary:
    Testing schedulers.py.
    Given a list of milestones and a list of learning rates, this function tests
    if the length of each list is the same. Furthermore, it will assess if
    the current learning rate (output from the function we are testing) is a float
    and corresponds to the expected learning rate given the milestones.

    Args:
        milestones: list of epochs indices (number of epochs)
        lr_list: learning rates list

    Returns:
        None

    Examples:
        input:
            milestones = [10,25,50]
            lr_list = [[0.00001],[0.000005],[0.000001]]
    """

    assert len(milestones) == len(lr_list)

    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=0.01)
    s = schedulers.LRListScheduler(optimizer, milestones=milestones, lr_list=lr_list)

    index_rng = range(milestones[0], milestones[1])
    for i in range((milestones[-1]) + 1):
        if i < milestones[0]:
            expected_lr = [0.01]
        elif i in index_rng:
            expected_lr = lr_list[0]
        else:
            expected_lr = lr_list[1]

        current_lr = s.get_lr()[0]
        assert s.get_lr() == expected_lr
        assert isinstance(current_lr, float)
        s.step()


@dataclass
class SchedulerTestConfig:
    cfg: dict
    init_lr: float
    expected_lrs: list[float]


TEST_SCHEDULERS = [
    SchedulerTestConfig(
        cfg=dict(
            type="LRListScheduler",
            params=dict(milestones=[2, 5], lr_list=[[0.5], [0.1]])
        ),
        init_lr=1.0,
        expected_lrs=[1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
    ),
    SchedulerTestConfig(
        cfg=dict(type="LRListScheduler", params=dict(milestones=[1], lr_list=[[0.1]])),
        init_lr=0.1,
        expected_lrs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ),
    SchedulerTestConfig(
        cfg=dict(type="LRListScheduler", params=dict(milestones=[1], lr_list=[[0.5]])),
        init_lr=0.1,
        expected_lrs=[0.1, 0.5, 0.5, 0.5],
    ),
]


@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_build_scheduler(test_cfg: SchedulerTestConfig) -> None:
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    s = schedulers.build_scheduler(test_cfg.cfg, optimizer)
    num_epochs = len(test_cfg.expected_lrs)
    for e in range(num_epochs):
        current_lrs = s.get_lr()
        print(f"Epoch {e}: LR={current_lrs}, expected={test_cfg.expected_lrs[e]}")
        for lr in current_lrs:
            assert lr == test_cfg.expected_lrs[e]
            assert isinstance(lr, float)
        s.step()


@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_resume_scheduler_after_each_epoch(test_cfg: SchedulerTestConfig) -> None:
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    s = schedulers.build_scheduler(test_cfg.cfg, optimizer)
    num_epochs = len(test_cfg.expected_lrs)
    for e in range(num_epochs):
        current_lrs = s.get_lr()
        print(f"Epoch {e}: LR={current_lrs}, expected={test_cfg.expected_lrs[e]}")
        for lr in current_lrs:
            assert lr == test_cfg.expected_lrs[e]
            assert isinstance(lr, float)
        s.step()

        optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
        new_scheduler = schedulers.build_scheduler(test_cfg.cfg, optimizer)
        schedulers.load_scheduler_state(new_scheduler, s.state_dict())

        # new_scheduler.load_state_dict(s.state_dict())
        #
        # # Update the learning rate for the optimizer based on the scheduler
        # resume_lrs = new_scheduler.get_last_lr()
        # for group, resume_lr in zip(optimizer.param_groups, resume_lrs):
        #     group['lr'] = resume_lr
        s = new_scheduler
