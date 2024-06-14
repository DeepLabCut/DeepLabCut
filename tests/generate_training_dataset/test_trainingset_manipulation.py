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
"""Tests for deeplabcut/generate_training_dataset/metadata.py"""
from __future__ import annotations

import pytest

import deeplabcut.generate_training_dataset.trainingsetmanipulation as trainingsetmanipulation


@pytest.mark.parametrize(
    "train_fraction", [1, 2, 5, 17, 24, 29, 34, 47, 50, 53, 61, 68, 75, 90, 95, 97, 99]
)
@pytest.mark.parametrize("n_train", [1, 2, 3, 5, 7, 11, 37, 62, 153])
@pytest.mark.parametrize("n_test", [1, 2, 3, 5, 7, 13, 19, 85, 112])
def test_compute_padding(train_fraction: int, n_train: int, n_test: int) -> None:
    """
    More complete tests can be run with:
        "train_fraction": list(range(1, 100))
        "n_train": list(range(1, 200))
        "n_test": list(range(1, 200))

    This was done locally, but as it's many many tests to run a subset was selected here
    """
    train_frac = train_fraction / 100
    train_pad, test_pad = trainingsetmanipulation._compute_padding(
        train_frac, n_train, n_test
    )
    print()
    print(train_fraction, n_train, n_test, train_pad, test_pad)
    frac = round((n_train + train_pad)/(n_train + n_test + train_pad + test_pad), 2)
    assert train_frac == frac
