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

from deeplabcut.core.config.config_mixin import ConfigMixin


@dataclass
class OptimizerConfig(ConfigMixin):
    """Optimizer configuration.

    Attributes:
        type: Optimizer type (e.g., AdamW, SGD)
        params: Optimizer parameters
    """

    type: str = ""
    params: dict[str, Any] | None = None


@dataclass
class SchedulerConfig(ConfigMixin):
    """Learning rate scheduler configuration.

    Attributes:
        type: Scheduler type (e.g., LRListScheduler, CosineAnnealingLR, SequentialLR)
        params: Scheduler parameters
    """

    type: str = ""
    params: dict[str, Any] | None = None


@dataclass
class SnapshotCheckpointConfig(ConfigMixin):
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
class RunnerConfig(ConfigMixin):
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
        resume_training_from: Path to resume training from
        load_weights_only: Value for torch.load() weights_only parameter
    """

    type: str = "PoseTrainingRunner"
    # TODO @deruyter92: Currently different configs for device are used in 
    # parallel. We should probably move to only 'PoseConfig.device'. This is 
    # kept here for backwards compatibility.
    gpus: Any | None = None
    device: str = "auto" # <- unused, but present in test scripts. 
    key_metric: str = "test.mAP"
    key_metric_asc: bool = True
    eval_interval: int = 10
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    snapshots: SnapshotCheckpointConfig | None = None
    resume_training_from: str | None = None
    load_weights_only: bool | None = None
