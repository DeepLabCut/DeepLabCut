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
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn as nn


ModelType = TypeVar("ModelType", bound=nn.Module)


class Runner(ABC, Generic[ModelType]):
    """Runner base class

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        device: str = "cpu",
        gpus: list[int] | None = None,
        snapshot_path: str | Path | None = None,
    ):
        """
        Args:
            model: the model to run
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            gpus: the list of GPU indices to use for multi-GPU training
            snapshot_path: the path of a snapshot from which to load model weights
        """
        if gpus is None:
            gpus = []

        if len(gpus) == 1:
            if device != "cuda":
                raise ValueError(
                    "When specifying a GPU index to train on, the device must be set "
                    f"to 'cuda'. Found {device}"
                )
            device = f"cuda:{gpus[0]}"

        self.model = model
        self.device = device
        self.snapshot_path = snapshot_path
        self._gpus = gpus
        self._data_parallel = len(gpus) > 1

    @staticmethod
    def load_snapshot(
        snapshot_path: str | Path,
        device: str,
        model: ModelType,
    ) -> dict:
        """Loads the state dict for a model from a file

        This method loads a file containing a DeepLabCut PyTorch model snapshot onto
        a given device, and sets the model weights using the state_dict.

        Args:
            snapshot_path: the path containing the model weights to load
            device: the device on which the model should be loaded
            model: the model for which the weights are loaded

        Returns:
            The content of the snapshot file.
        """
        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot["model"])
        return snapshot
