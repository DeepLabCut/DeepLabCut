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

from deeplabcut.utils.plotting import PLOT_NAMES, PlottingResults, _plot_trajectories

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


@pytest.mark.parametrize(
    "individuals, expected_order",
    [
        pytest.param(["mouse2", "mouse10", "mouse1"], ["mouse2", "mouse10", "mouse1"], id="project-order"),
        pytest.param(None, ["mouse1", "mouse2", "mouse10"], id="data-order"),
    ],
)
def test_plot_trajectories_keeps_the_order_of_individuals(tmp_path, individuals, expected_order):
    columns = pd.MultiIndex.from_product(
        [["scorer"], ["mouse1", "mouse2", "mouse10"], BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    h5file = tmp_path / "videoDLC_scorer.h5"
    pd.DataFrame(0.0, index=range(10), columns=columns).to_hdf(h5file, key="df_with_missing")

    figures = _plot_trajectories(h5file, BODYPARTS, individuals, dest_folder=str(tmp_path))

    first_figure_per_individual = list(figures)[:: len(PLOT_NAMES)]
    assert [path.name for path in first_figure_per_individual] == [f"trajectory{i}.png" for i in expected_order]
