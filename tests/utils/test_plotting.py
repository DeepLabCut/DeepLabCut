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
"""Tests for ``deeplabcut.utils.plotting``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from deeplabcut.utils.plotting import PlottingResults

BODYPARTS = ["nose", "tail"]


@pytest.fixture
def open_plt_figure():
    """A pyplot figure standing in for one another caller is still showing."""
    fig = plt.figure()
    yield fig
    plt.close(fig)


def test_plotting_results_without_showing_leaves_pyplot_untouched(tmp_path, open_plt_figure):
    # Synthetic predictions: 10 frames of zeros
    columns = pd.MultiIndex.from_product([BODYPARTS, ["x", "y", "likelihood"]], names=["bodyparts", "coords"])
    predictions = pd.DataFrame(0.0, index=range(10), columns=columns)
    cfg = {"pcutoff": 0.1, "colormap": "viridis", "alphavalue": 0.7}
    figures_before = plt.get_fignums()  # prevent left overs by other tests

    figures = PlottingResults(tmp_path, predictions, cfg, BODYPARTS, "", showfigures=False)

    assert [p.name for p in figures] == ["trajectory.png", "plot.png", "plot-likelihood.png", "hist.png"]
    assert all(p.is_file() for p in figures)
    assert all(isinstance(fig, Figure) for fig in figures.values())
    assert plt.get_fignums() == figures_before
