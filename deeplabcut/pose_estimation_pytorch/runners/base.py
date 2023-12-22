#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn as nn


ModelType = TypeVar("ModelType", bound=nn.Module)


class Task(Enum):
    """A task to solve"""

    BOTTOM_UP = "BU"
    DETECT = "DT"
    TOP_DOWN = "TD"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            for member in cls:
                if member.value == value:
                    return member
        return None


class Runner(ABC, Generic[ModelType]):
    """Runner base class

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        device: str = "cpu",
        snapshot_path: str | Path | None = None,
    ):
        """
        Args:
            model: the model to run
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            snapshot_path: the path of a snapshot from which to load model weights
        """
        self.model = model
        self.device = device
        self.snapshot_path = snapshot_path

    @staticmethod
    def load_snapshot(
        snapshot_path: str,
        device: str,
        model: ModelType,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """
        Args:
            snapshot_path: the path containing the model weights to load
            device: the device on which to load the model
            model: the model for which the weights are loaded
            optimizer: if defined, the optimizer weights to load

        Returns:
            the number of epochs the model was trained for
        """
        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(snapshot["optimizer_state_dict"])

        return snapshot["epoch"]
