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
"""Training configuration classes for DeepLabCut pose estimation models."""

from typing import Any

from pydantic import field_validator, model_validator

from deeplabcut.core.config import DLCBaseConfig
from deeplabcut.core.config.validation import (
    reject_multiprocessing_only_kwargs,
    reject_reserved_keys,
)
from deeplabcut.core.weight_init import WeightInitialization

# DataLoader arguments set by DeepLabCut, mapped to the option that controls them
_RESERVED_DATALOADER_KWARGS = {
    "dataset": None,
    "batch_size": "batch_size",
    "shuffle": None,
    "collate_fn": "data.collate",
    "num_workers": "dataloader_workers",
    "pin_memory": "dataloader_pin_memory",
}


class TrainSettingsConfig(DLCBaseConfig):
    """Training settings configuration.

    Attributes:
        batch_size: Training batch size
        dataloader_workers: Number of data loader workers
        dataloader_pin_memory: Whether to pin memory in data loader
        dataloader_kwargs: Additional keyword arguments for the training
            ``torch.utils.data.DataLoader`` (e.g. ``persistent_workers``). Arguments
            that DeepLabCut sets itself are reserved and cannot be given here.
        display_iters: Display interval for training progress
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        weight_init: Weight initialization configuration
    """

    batch_size: int = 8
    dataloader_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_kwargs: dict[str, Any] | None = None
    display_iters: int = 500
    epochs: int = 200
    seed: int = 42
    # @TODO @deruyter92 2026-02-13: V0 pipeline uses None for default initialization
    # (with ImageNet weights). We should update this to explicit WeightInitialization.
    weight_init: WeightInitialization | None = None

    @field_validator("dataloader_kwargs")
    @classmethod
    def validate_dataloader_kwargs(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        reject_reserved_keys(v, _RESERVED_DATALOADER_KWARGS, name="dataloader_kwargs")
        return v

    @model_validator(mode="after")
    def validate_dataloader_kwargs_against_workers(self) -> "TrainSettingsConfig":
        reject_multiprocessing_only_kwargs(
            self.dataloader_kwargs,
            self.dataloader_workers,
            name="dataloader_kwargs",
        )
        return self
