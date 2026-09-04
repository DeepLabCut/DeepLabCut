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

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import field_validator, model_validator

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

    Notes:
        Different config versions might have top-level WandB options instead of
        wandb_kwargs. Unknown top-level fields are scooped into wandb_kwargs.

    Attributes:
        type: Logger type (should be 'WandbLogger')
        project_name: The name of the wandb project
        run_name: The name of the wandb run
        image_log_interval: How often train/test images are logged in epochs
            (if None, train/test inputs are never logged)
        model: The model architecture to log
        train_folder: The path of the folder containing training files.
        wandb_kwargs: Additional keyword arguments to pass to wandb.init
    """

    # These options are reserved since they are explicitly passed to wandb.init (from `project_name` and `run_name`)
    _FORBIDDEN_WANDB_KWARGS: ClassVar[frozenset[str]] = frozenset({"project", "name"})

    type: Literal[LoggerType.WandbLogger]
    project_name: str = "deeplabcut"
    run_name: str = "tmp"
    image_log_interval: int | None = None
    model: dict | None = None
    train_folder: str | None = None
    wandb_kwargs: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def scoop_extra_wandb_kwargs(cls, data: Any) -> Any:
        """Move unknown top-level keys into ``wandb_kwargs``"""
        if not isinstance(data, dict):
            return data

        data = dict(data)
        known = set(cls.model_fields) | set(cls._alias_map())
        provided = set(data)
        extras = {key: data.pop(key) for key in provided - known}
        if not extras:
            return data

        nested = data.get("wandb_kwargs") or {}
        if not isinstance(nested, dict):
            raise ValueError("wandb_kwargs must be a dictionary when provided")

        overlap = set(extras) & set(nested)
        if overlap:
            raise ValueError(f"Duplicate wandb.init options at top-level and in wandb_kwargs: {sorted(overlap)}")

        data["wandb_kwargs"] = {**nested, **extras}
        return data

    @field_validator("wandb_kwargs")
    @classmethod
    def validate_wandb_kwargs(cls, v: dict | None) -> dict | None:
        """Reject nested keys that collide with reserved wandb.init options."""
        if not v:
            return v
        if set(v) & cls._FORBIDDEN_WANDB_KWARGS:
            raise ValueError(
                f"wandb_kwargs cannot include any of the reserved options: {sorted(cls._FORBIDDEN_WANDB_KWARGS)}"
            )


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
