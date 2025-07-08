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
import os
import pickle
from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn

ModelType = TypeVar("ModelType", bound=nn.Module)

_load_weights_only: bool = (
    os.getenv("TORCH_LOAD_WEIGHTS_ONLY", "true").lower() in ("true", "1")
)


def get_load_weights_only() -> bool:
    """Gets the default value to use when loading snapshots with `torch.load(...)`.

    Returns:
        The default `weights_only` value when loading snapshots using `torch.load(...)`.
    """
    global _load_weights_only
    return _load_weights_only


def set_load_weights_only(value: bool) -> None:
    """Sets the default value to use when loading snapshots with `torch.load(...)`.

    Args:
        value: The default `weights_only` value to use when loading snapshots using
            `torch.load(...)`.
    """
    global _load_weights_only
    _load_weights_only = value


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
        weights_only: bool | None = None,
    ) -> dict:
        """Loads the state dict for a model from a file

        This method loads a file containing a DeepLabCut PyTorch model snapshot onto
        a given device, and sets the model weights using the state_dict.

        Args:
            snapshot_path: The path containing the model weights to load
            device: The device on which the model should be loaded
            model: The model for which the weights are loaded
            weights_only: Value for torch.load() `weights_only` parameter.
                If False, the python pickle module is used implicitly, which is known to
                be insecure. Only set to False if you're loading data that you trust
                (e.g. snapshots that you created yourself). For more information, see:
                    https://pytorch.org/docs/stable/generated/torch.load.html
                If None, the default value is used:
                    `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`

        Returns:
            The content of the snapshot file.
        """
        snapshot = attempt_snapshot_load(snapshot_path, device, weights_only)
        
        # Handle the case where snapshot keys have 'model.' prefix
        snapshot_weights = snapshot["model"]
        model_state_dict = model.state_dict()
        
        # Diagnostic: Always add 'model.' prefix for superanimal_topviewmouse detectors
        is_topviewmouse = hasattr(model, 'superanimal_name') and getattr(model, 'superanimal_name', None) == 'superanimal_topviewmouse'
        is_detector = 'FasterRCNN' in str(type(model)) or 'SSDLite' in str(type(model))
        if is_topviewmouse and is_detector:
            print(f"DEBUG: Forcing prefix ADD for superanimal_topviewmouse detector!")
            cleaned_weights = {}
            for key, value in snapshot_weights.items():
                if not key.startswith('model.'):
                    cleaned_key = 'model.' + key  # Add 'model.' prefix
                    cleaned_weights[cleaned_key] = value
                else:
                    cleaned_weights[key] = value
            print(f"DEBUG: Loading cleaned weights with {len(cleaned_weights)} keys")
            model.load_state_dict(cleaned_weights)
        elif (any(key.startswith('model.') for key in snapshot_weights.keys()) and 
            not any(key.startswith('model.') for key in model_state_dict.keys())):
            print(f"DEBUG: Detected 'model.' prefix mismatch, cleaning keys...")
            # Strip the 'model.' prefix from snapshot keys
            cleaned_weights = {}
            for key, value in snapshot_weights.items():
                if key.startswith('model.'):
                    cleaned_key = key[6:]  # Remove 'model.' prefix
                    cleaned_weights[cleaned_key] = value
                else:
                    cleaned_weights[key] = value
            print(f"DEBUG: Loading cleaned weights with {len(cleaned_weights)} keys")
            model.load_state_dict(cleaned_weights)
        else:
            print(f"DEBUG: No prefix mismatch, loading original weights")
            # Use original snapshot weights
            model.load_state_dict(snapshot["model"])
        
        return snapshot


def attempt_snapshot_load(
    path: str | Path,
    device: str,
    weights_only: bool | None = None,
) -> dict:
    """Attempts to load a snapshot using `torch.load(...)`.

    Args:
        path: The path of the snapshot to try to load..
        device: The device to use for the `map_location`.
        weights_only: Value for torch.load() `weights_only` parameter.
            If False, the python pickle module is used implicitly, which is known to be
            insecure. Only set to False if you're loading data that you trust (e.g.
            snapshots that you created yourself). For more information, see:
                https://pytorch.org/docs/stable/generated/torch.load.html
            If None, the default value is used:
                `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`

    Returns:
        The loaded snapshot.

    Raises:
        pickle.UnpicklingError: If `weights_only=True` but the snapshot failed to load
            with `weights_only=True`.
    """
    try:
        if weights_only is None:
            weights_only = get_load_weights_only()

        snapshot = torch.load(path, map_location=device, weights_only=weights_only)
    except pickle.UnpicklingError as err:
        logging.error(
            f"\nFailed to load the snapshot: {path}.\n\n"
            "If you trust the snapshot that you're trying to load, you can try\n"
            "calling `Runner.load_snapshot` with `weights_only=False`. See the \n"
            "error message below for more information and warnings.\n"
            "You can set the `weights_only` parameter in the model configuration (\n"
            "the content of the pytorch_config.yaml), as:\n\n```\n"
            "runner:\n"
            "  load_weights_only: False\n```\n\n"
            "If it's the detector snapshot that's failing to load, place the\n"
            "`load_weights_only` key under the detector runner:\n\n```\n"
            "detector:\n"
            "    runner:\n"
            "      load_weights_only: False\n```\n\n"
            "You can also set the default `load_weights_only` that will be used when\n"
            "the `load_weights_only` variable is not set in the `pytorch_config.yaml`\n"
            "using `deeplabcut.pose_estimation_pytorch.set_load_weights_only(value)`:\n"
            "\n```\n"
            "from deeplabcut.pose_estimation_pytorch import set_load_weights_only\n"
            "set_load_weights_only(True)\n"
            "```\n\n"
            "You can also set the value for `load_weights_only` with a \n"
            "`TORCH_LOAD_WEIGHTS_ONLY` environment variable. If you call \n"
            "`TORCH_LOAD_WEIGHTS_ONLY=False python -m deeplabcut`, it will launch the\n"
            "DeepLabCut GUI with the default `load_weights_only` value to False.\n"
            "If you set this value to `False`, make sure you only load snapshots that\n"
            "you trust.\n\n"
        )
        raise err

    return snapshot


def fix_snapshot_metadata(path: str | Path) -> None:
    """Replace numpy floats in snapshot metrics

    Only call this method with snapshots that you trust, as torch.load(...) is called
    with `weights_only=False`. For more information, see:
        https://pytorch.org/docs/stable/generated/torch.load.html

    DeepLabCut PyTorch snapshots trained with older releases may have `numpy` floats in
    the stored metrics. This method opens the snapshots (with `weights_only=False`),
    replaces the numpy floats with python floats (allowing to load with
    `weights_only=True`), and saves the new snapshot data.

    Warning: This overwrites your existing snapshot. If you want to ensure that no data
    is lost, copy your snapshot before calling `fix_snapshot_metadata`.

    Args:
        path: The path of the snapshot to fix.
    """
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    metrics = snapshot.get("metadata", {}).get("metrics")
    if metrics is not None:
        snapshot["metadata"]["metrics"] = {k: float(v) for k, v in metrics.items()}

    torch.save(snapshot, path)


def _add_numpy_to_torch_safe_globals():
    """
    Attempts tot add numpy classes allowing snapshots containing numpy floats in the
    metrics to be loaded without needing to change the `weights_only` argument.

    This fix only works for `numpy>=1.25.0`.
    """
    try:
        from numpy.core.multiarray import scalar
        from numpy.dtypes import Float64DType
        torch.serialization.add_safe_globals([np.dtype, Float64DType, scalar])
    except Exception:
        pass


_add_numpy_to_torch_safe_globals()
