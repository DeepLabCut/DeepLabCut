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

import os
import random
from pathlib import Path

import numpy as np
import torch

from deeplabcut.utils.auxiliaryfunctions import read_plainconfig


def create_folder(path_to_folder):
    """Creates all folders contained in the path.

    Args:
        path_to_folder: Path to the folder that should be created
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def fix_seeds(seed: int) -> None:
    """
    Fixes the random seed for python, numpy and pytorch

    Args:
        seed: the seed to set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(model_config: dict) -> str:
    """Determines which device should be used from the model config

    When the device is set to 'auto':
        If an Nvidia GPU is available, selects the device as cuda:0.
        Selects 'mps' if available (on macOS) and the net type is compatible.
        Otherwise, returns 'cpu'.
    Otherwise, simply returns the selected device

    Args:
        model_config: the configuration for the pose model

    Returns:
        the device on which training should be run
    """
    device = model_config["device"]
    supports_mps = "resnet" in model_config.get("net_type", "resnet")

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif supports_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
