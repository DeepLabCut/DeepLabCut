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

import logging
import pickle
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
        weights_only: bool = True,
    ) -> dict:
        """Loads the state dict for a model from a file

        This method loads a file containing a DeepLabCut PyTorch model snapshot onto
        a given device, and sets the model weights using the state_dict.

        Args:
            snapshot_path: The path containing the model weights to load
            device: The device on which the model should be loaded
            model: The model for which the weights are loaded
            weights_only: Value for torch.load() `weights_only` parameter. If False, the
                python pickle module is used implicitly, which is known to be insecure.
                Only set to False if you're loading data that you trust (e.g. snapshots
                that you created yourself). For more information, see:
                    https://pytorch.org/docs/stable/generated/torch.load.html

        Returns:
            The content of the snapshot file.
        """
        try:
            snapshot = torch.load(
                snapshot_path,
                map_location=device,
                weights_only=weights_only,
            )
        except pickle.UnpicklingError as err:
            print(
                f"\nFailed to load the snapshot: {snapshot_path}.\n"
                "If you trust the snapshot that you're trying to load, you can try "
                "calling `Runner.load_snapshot` with `weights_only=False`. See "
                "the message below for more information and warnings.\n"
                "You can set the `weights_only` parameter in the model configuration ("
                "the content of the pytorch_config.yaml), as:\n```\n"
                "runner:\n"
                "  load_weights_only: False\n```\n"
            )
            raise err

        model.load_state_dict(snapshot["model"])
        return snapshot
