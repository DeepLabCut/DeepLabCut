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

from pydantic.dataclasses import dataclass
from dataclasses import field
from deeplabcut.core.weight_init import WeightInitialization


@dataclass
class TrainSettingsConfig:
    """Training settings configuration.

    Attributes:
        batch_size: Training batch size
        dataloader_workers: Number of data loader workers
        dataloader_pin_memory: Whether to pin memory in data loader
        display_iters: Display interval for training progress
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        weight_init: Weight initialization configuration
    """

    batch_size: int = 8
    dataloader_workers: int = 0
    dataloader_pin_memory: bool = False
    display_iters: int = 500
    epochs: int = 200
    seed: int = 42
    weight_init: WeightInitialization = field(default_factory=WeightInitialization)
