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
"""Runner configuration class for DeepLabCut pose estimation models."""

from pydantic.dataclasses import dataclass

from typing import Any


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        type: Optimizer type (e.g., AdamW, SGD)
        params: Optimizer parameters
    """

    type: str = ""
    params: dict[str, Any] | None = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        type: Scheduler type (e.g., LRListScheduler, CosineAnnealingLR, SequentialLR)
        params: Scheduler parameters
    """

    type: str = ""
    params: dict[str, Any] | None = None


@dataclass
class SnapshotCheckpointConfig:
    """Snapshot configuration for model checkpoints.

    Attributes:
        max_snapshots: Maximum number of snapshots to keep
        save_epochs: Interval for saving snapshots
        save_optimizer_state: Whether to save optimizer state
    """

    max_snapshots: int = 5
    save_epochs: int = 25
    save_optimizer_state: bool = False


@dataclass
class RunnerConfig:
    """Training runner configuration.

    Attributes:
        type: Runner type (e.g., PoseTrainingRunner)
        gpus: GPU configuration
        key_metric: Key metric for evaluation
        key_metric_asc: Whether key metric should be ascending
        eval_interval: Evaluation interval in epochs
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        snapshots: Snapshot configuration
    """

    type: str = "PoseTrainingRunner"
    gpus: Any | None = None
    key_metric: str = "test.mAP"
    key_metric_asc: bool = True
    eval_interval: int = 10
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    snapshots: SnapshotCheckpointConfig | None = None
    resume_training_from: str | None = None
