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
import pytest

import numpy as np
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.runners.schedulers import build_scheduler


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
    scheduler = build_scheduler(data["config"], optimizer)

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
