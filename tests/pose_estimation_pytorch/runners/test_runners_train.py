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
from dataclasses import dataclass
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners.schedulers as schedulers
import deeplabcut.pose_estimation_pytorch.runners.train as train_runners


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


@patch("deeplabcut.pose_estimation_pytorch.runners.train.CSVLogger", Mock())
@pytest.mark.parametrize("runner_cls", [
    train_runners.PoseTrainingRunner, train_runners.DetectorTrainingRunner,
])
@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_training_with_scheduler(runner_cls, test_cfg: SchedulerTestConfig) -> None:
    runner = _fit_runner_and_check_lrs(
        runner_cls,
        test_cfg.init_lr,
        test_cfg.cfg,
        test_cfg.expected_lrs,
    )
    assert runner.current_epoch == len(test_cfg.expected_lrs)


@patch("deeplabcut.pose_estimation_pytorch.runners.train.CSVLogger", Mock())
@pytest.mark.parametrize("runner_cls", [
    train_runners.PoseTrainingRunner, train_runners.DetectorTrainingRunner,
])
@pytest.mark.parametrize("test_cfg", TEST_SCHEDULERS)
def test_resuming_training_scheduler_every_epoch(
    runner_cls,
    test_cfg: SchedulerTestConfig,
):
    snapshot_to_load = None
    for epoch, expected_lr in enumerate(test_cfg.expected_lrs):
        runner = _fit_runner_and_check_lrs(
            runner_cls,
            test_cfg.init_lr,
            test_cfg.cfg,
            [expected_lr],  # trains for 1 epoch
            snapshot_to_load=snapshot_to_load,
        )
        snapshot_to_load = dict(
            metadata=dict(epoch=epoch + 1), scheduler=runner.scheduler.state_dict()
        )


@patch("deeplabcut.pose_estimation_pytorch.runners.train.CSVLogger", Mock())
@pytest.mark.parametrize("runner_cls", [
    train_runners.PoseTrainingRunner, train_runners.DetectorTrainingRunner,
])
@pytest.mark.parametrize(
    "test_cfg, resume_epoch",
    [
        (
            SchedulerTestConfig(
                cfg=dict(
                    type="LRListScheduler",
                    params=dict(milestones=[2, 5], lr_list=[[0.5], [0.1]])
                ),
                init_lr=1.0,
                expected_lrs=[1.0, 1.0, 0.5, 1.0, 1.0, 0.1, 0.1, 0.1],
            ),
            3,  # cut after the 3rd epoch - restart at LR=1 until epoch 5
        ),
        (
            SchedulerTestConfig(
                cfg=dict(type="StepLR", params=dict(step_size=4, gamma=0.1)),
                init_lr=1.0,
                expected_lrs=(4 * [1.0]) + (4 * [0.1]) + (4 * [0.01]) + (4 * [0.001]),
            ),
            3,  # cut after the 3rd epoch - restart at LR=1 and update at 4 correctly
        ),
        (
            SchedulerTestConfig(
                cfg=dict(type="StepLR", params=dict(step_size=4, gamma=0.1)),
                init_lr=1.0,
                expected_lrs=(4 * [1.0]) + [0.1, 1, 1, 1] + (4 * [0.1]),
            ),
            5,  # cut after the 5th epoch - restart at LR=1 and update again at 8
        )
    ]
)
def test_resuming_training_with_no_scheduler_state(
    runner_cls,
    test_cfg: SchedulerTestConfig,
    resume_epoch: int
):
    """
    Without a scheduler config, there is no way to set the initial LR. All we can do is
    set the last_epoch value, and adjust correctly at milestones going forward.
    """
    runner = _fit_runner_and_check_lrs(
        runner_cls,
        test_cfg.init_lr,
        test_cfg.cfg,
        test_cfg.expected_lrs[:resume_epoch],
    )
    assert runner.current_epoch == resume_epoch

    runner = _fit_runner_and_check_lrs(
        runner_cls,
        test_cfg.init_lr,
        test_cfg.cfg,
        expected_lrs=test_cfg.expected_lrs[resume_epoch:],
        snapshot_to_load=dict(metadata=dict(epoch=resume_epoch)),
    )
    assert runner.current_epoch == len(test_cfg.expected_lrs)


def _fit_runner_and_check_lrs(
    runner_cls,
    init_lr: float,
    scheduler_cfg: dict,
    expected_lrs: list[float],
    snapshot_to_load: dict | None = None,
) -> train_runners.TrainingRunner:
    runner_kwargs = dict(device="cpu", eval_interval=1_000_000)
    optimizer = torch.optim.SGD([torch.randn(2, 2)], lr=init_lr)
    scheduler = schedulers.build_scheduler(scheduler_cfg, optimizer)
    num_epochs = len(expected_lrs)

    with patch(
        "deeplabcut.pose_estimation_pytorch.runners.Runner.load_snapshot"
    ) as mock_load_snapshot:
        snapshot_path = None
        mock_load_snapshot.return_value = dict()
        if snapshot_to_load is not None:
            snapshot_path = "fake_snapshot.pt"
            mock_load_snapshot.return_value = snapshot_to_load

        print()
        print(f"Scheduler: {scheduler}")
        print(f"Starting training for {num_epochs} epochs")
        runner = runner_cls(
            model=Mock(),
            optimizer=optimizer,
            snapshot_manager=Mock(),
            scheduler=scheduler,
            snapshot_path=snapshot_path,
            **runner_kwargs
        )

        # Mock the step call; check that the learning rate is correct for the epoch
        def step(*args, **kwargs):
            # the current_epoch value is indexed at 1
            total_epoch = (runner.current_epoch - 1)
            epoch = total_epoch - runner.starting_epoch
            _assert_learning_rates_match(total_epoch, optimizer, expected_lrs[epoch])
            optimizer.step()
            return dict(total_loss=0)

        train_loader, val_loader = [Mock()], [Mock()]
        runner.step = step
        runner.fit(train_loader, val_loader, epochs=num_epochs, display_iters=1000)

    return runner


def _assert_learning_rates_match(e, optimizer, expected):
    current_lrs = [g["lr"] for g in optimizer.param_groups]
    print(f"Epoch {e}: LR={current_lrs}, expected={expected}")
    for lr in current_lrs:
        assert isinstance(lr, float)
        np.testing.assert_almost_equal(lr, expected)
