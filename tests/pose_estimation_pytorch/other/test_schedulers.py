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

import pytest
import torch
from torch.optim import SGD

import deeplabcut.pose_estimation_pytorch.runners.schedulers as deeplabcut_torch_schedulers


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


milestones = random.sample(range(0, 999), 2)
milestones.sort()
data = [([10, 430], [[0.05], [0.005]]), (milestones, generate_random_lr_list(2))]
# testing for default values in pytorch_config and also for random values with pytest parametrize


@pytest.mark.parametrize("milestones, lr_list", data)
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
    lrlistscheduler = deeplabcut_torch_schedulers.LRListScheduler(
        optimizer, milestones=milestones, lr_list=lr_list
    )

    index_rng = range(milestones[0], milestones[1])
    for i in range((milestones[-1]) + 1):
        if i < milestones[0]:
            expected_lr = [0.01]
        elif i in index_rng:
            expected_lr = lr_list[0]
        else:
            expected_lr = lr_list[1]

        current_lr = lrlistscheduler.get_lr()[0]
        assert lrlistscheduler.get_lr() == expected_lr
        assert isinstance(current_lr, float)
        lrlistscheduler.step()
