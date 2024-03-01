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

import abc
import os
import random

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


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq: The sequence to be checked.
        expected_type: Expected type of sequence items.
        seq_type: Expected sequence type.
    Returns:
        Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def get_pytorch_config(modelfolder):
    pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
    pytorch_cfg = read_plainconfig(pytorch_config_path)

    return pytorch_cfg
