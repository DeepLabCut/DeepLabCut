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
"""Logger configuration classes for DeepLabCut training runs."""

from enum import Enum
from typing import Literal

from deeplabcut.core.config import DLCBaseConfig


class LoggerType(str, Enum):
    WandbLogger = "WandbLogger"
    CSVLogger = "CSVLogger"


class LoggerConfig(DLCBaseConfig):
    """Base configuration for all loggers.

    Attributes:
        type: The type of logger to use (WandbLogger or CSVLogger)
    """

    type: str


class WandbLoggerConfig(LoggerConfig):  #
    """Configuration for Weights & Biases (wandb) logger.

    This logger tracks experiments and logs data to Weights & Biases.
    Refer to: https://docs.wandb.ai/guides for more information.

    Attributes:
        type: Logger type (should be 'WandbLogger')
        project_name: The name of the wandb project
        run_name: The name of the wandb run
        entity: The wandb user or team under which the run is logged
        notes: A longer description of the run, displayed in the wandb UI
        tags: Tags for the run, used to organize and filter runs in the wandb UI
        group: The name of the group to which this run belongs
        job_type: The type of job being logged (e.g. 'train' or 'eval')
        image_log_interval: How often train/test images are logged in epochs
            (if None, train/test inputs are never logged)
        model: The model architecture to log
        train_folder: The path of the folder containing training files.
        wandb_kwargs: Additional keyword arguments to pass to wandb.init. Use this for
            wandb.init parameters which are not declared above.
    """

    type: Literal[LoggerType.WandbLogger]
    project_name: str = "deeplabcut"
    run_name: str = "tmp"
    entity: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    group: str | None = None
    job_type: str | None = None
    image_log_interval: int | None = None
    model: dict | None = None
    train_folder: str | None = None
    wandb_kwargs: dict | None = None


class CSVLoggerConfig(LoggerConfig):  #
    """Configuration for CSV logger.

    This logger saves training stats and metrics to a CSV file.

    Attributes:
        type: Logger type (should be 'CSVLogger')
        train_folder: The path of the folder containing training files.
        log_filename: The name of the file in which to store training stats
    """

    type: Literal[LoggerType.CSVLogger]
    train_folder: str = ""
    log_filename: str = "learning_stats.csv"
