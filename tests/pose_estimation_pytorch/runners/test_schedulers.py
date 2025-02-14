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
"""Tests building schedulers from config"""
import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn

import deeplabcut.pose_estimation_pytorch.runners.schedulers as schedulers


def generate_random_lr_list(num_floats: int):
    """Generate list of lists including random numbers.

    Args:
        num_floats: number of floats we want to include in our list

    Returns:
        ran_list: random list of sorted numbers, being first number bigger than the last
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
    """Testing schedulers.py.

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
        optimizer.step()
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
    SchedulerTestConfig(
        cfg=dict(type="StepLR", params=dict(step_size=3, gamma=0.1)),
        init_lr=1.0,
        expected_lrs=[1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001],
    ),
]


@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_build_scheduler(test_cfg: SchedulerTestConfig) -> None:
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    s = schedulers.build_scheduler(test_cfg.cfg, optimizer)
    print()
    print(f"Scheduler: {s}")
    num_epochs = len(test_cfg.expected_lrs)
    for e in range(num_epochs):
        _assert_learning_rates_match(e, optimizer, test_cfg.expected_lrs[e])
        optimizer.step()
        s.step()


@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_resume_scheduler_after_each_epoch(test_cfg: SchedulerTestConfig) -> None:
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    s = schedulers.build_scheduler(test_cfg.cfg, optimizer)
    print()
    print(f"Scheduler: {s}")
    num_epochs = len(test_cfg.expected_lrs)
    for e in range(num_epochs):
        _assert_learning_rates_match(e, optimizer, test_cfg.expected_lrs[e])
        optimizer.step()
        s.step()

        optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
        new_scheduler = schedulers.build_scheduler(test_cfg.cfg, optimizer)
        schedulers.load_scheduler_state(new_scheduler, s.state_dict())
        s = new_scheduler


@pytest.mark.parametrize(
    "test_cfg, middle_epoch",
    [
        (TEST_SCHEDULERS[0], 3),
        (TEST_SCHEDULERS[1], 5),
        (TEST_SCHEDULERS[2], 2),
        (TEST_SCHEDULERS[3], 2),
        (TEST_SCHEDULERS[3], 3),
        (TEST_SCHEDULERS[3], 4),
    ],
)
def test_two_stage_training(test_cfg: SchedulerTestConfig, middle_epoch: int) -> None:
    num_epochs = len(test_cfg.expected_lrs)
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    s = schedulers.build_scheduler(test_cfg.cfg, optimizer)

    print()
    print(f"Scheduler: {s}")
    for e in range(middle_epoch):
        _assert_learning_rates_match(e, optimizer, test_cfg.expected_lrs[e])
        optimizer.step()
        s.step()

    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=test_cfg.init_lr)
    new_scheduler = schedulers.build_scheduler(test_cfg.cfg, optimizer)
    schedulers.load_scheduler_state(new_scheduler, s.state_dict())
    s = new_scheduler
    for e in range(middle_epoch, num_epochs):
        _assert_learning_rates_match(e, optimizer, test_cfg.expected_lrs[e])
        s.step()


@pytest.mark.parametrize(
    "data",
    [
        dict(  # example with 3 warm-up epochs
            config=dict(
                dict(
                    type="ConstantLR",
                    params=dict(factor=0.1, total_iters=3),
                ),
            ),
            start_lr=1.0,
            expected_lrs=[[0.1], [0.1], [0.1], [1.0], [1.0]],
        ),
        dict(  # example from torch.optim.lr_scheduler.SequentialLR
            config=dict(
                type="SequentialLR",
                params=dict(
                    schedulers=[
                        dict(
                            type="ConstantLR",
                            params=dict(factor=0.1, total_iters=2),
                        ),
                        dict(type="ExponentialLR", params=dict(gamma=0.9)),
                    ],
                    milestones=[2],
                ),
            ),
            start_lr=1.0,
            expected_lrs=[[0.1], [0.1], [1.0], [0.9], [0.81], [0.729]],
        ),
        dict(  # example from torch.optim.lr_scheduler.SequentialLR
            config=dict(
                type="SequentialLR",
                params=dict(
                    schedulers=[
                        dict(
                            type="ConstantLR",
                            params=dict(factor=0.1, total_iters=2),
                        ),
                        dict(type="StepLR", params=dict(step_size=2, gamma=0.1)),
                    ],
                    milestones=[5],
                ),
            ),
            start_lr=1.0,
            expected_lrs=[
                [0.1], [0.1], [1.0], [1.0], [1.0],  # ConstantLR
                [1.0], [1.0], [0.1], [0.1], [0.01],  # StepLR
            ],
        ),
    ],
)
def test_build_sequential_lr(data):
    print("\nTESTING")
    start_lr = data["start_lr"]
    print(f"Start LR: {start_lr}")
    model = nn.Linear(in_features=1, out_features=1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=start_lr)

    print("BUILDING")
    scheduler = schedulers.build_scheduler(data["config"], optimizer)

    print("RUNNING")
    lrs = []
    for epoch in range(len(data["expected_lrs"])):
        lrs.append(scheduler.get_last_lr())
        print(scheduler.get_last_lr())
        scheduler.step()

    print(f"Expected: {data['expected_lrs']}")
    print(f"Actual: {lrs}")
    np.testing.assert_allclose(
        np.asarray(data["expected_lrs"]),
        np.asarray(lrs),
        atol=1e-10,
    )


def _assert_learning_rates_match(e, optimizer, expected):
    current_lrs = [g["lr"] for g in optimizer.param_groups]
    print(f"Epoch {e}: LR={current_lrs}, expected={expected}")
    for lr in current_lrs:
        assert isinstance(lr, float)
        np.testing.assert_almost_equal(lr, expected)
