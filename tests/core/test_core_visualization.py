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
"""Tests for deeplabcut/core/visualization.py."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

import deeplabcut.core.visualization as viz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 20, 30
N_BODYPARTS = 3
N_PAF_FIELDS = 2


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to avoid memory leaks."""
    yield
    plt.close("all")


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def _scmap() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random(size=(H, W)).astype(np.float32)


# ---------------------------------------------------------------------------
# form_figure
# ---------------------------------------------------------------------------


def test_form_figure_returns_figure_and_axes():
    fig, ax = viz.form_figure(W, H)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_form_figure_sets_x_limits():
    _, ax = viz.form_figure(W, H)
    assert ax.get_xlim() == (0, W)


def test_form_figure_sets_y_limits_inverted():
    _, ax = viz.form_figure(W, H)
    bottom, top = ax.get_ylim()
    # invert_yaxis makes bottom > top
    assert bottom > top


def test_form_figure_axis_is_off():
    _, ax = viz.form_figure(W, H)
    assert not ax.axison


# ---------------------------------------------------------------------------
# visualize_scoremaps
# ---------------------------------------------------------------------------


def test_visualize_scoremaps_returns_figure_and_axes():
    fig, ax = viz.visualize_scoremaps(_image(), _scmap())
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_visualize_scoremaps_renders_two_images():
    _, ax = viz.visualize_scoremaps(_image(), _scmap())
    # imshow adds AxesImage artists; one for the image, one for the scoremap overlay
    images = ax.get_images()
    assert len(images) == 2


def test_visualize_scoremaps_scoremap_has_alpha():
    _, ax = viz.visualize_scoremaps(_image(), _scmap())
    scmap_artist = ax.get_images()[1]
    assert scmap_artist.get_alpha() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# visualize_locrefs
# ---------------------------------------------------------------------------


def _locref() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    locref_x = rng.uniform(-5, 5, size=(H, W)).astype(np.float32)
    locref_y = rng.uniform(-5, 5, size=(H, W)).astype(np.float32)
    return locref_x, locref_y


def test_visualize_locrefs_returns_figure_and_axes():
    locref_x, locref_y = _locref()
    fig, ax = viz.visualize_locrefs(_image(), _scmap(), locref_x, locref_y)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_visualize_locrefs_adds_quiver():
    locref_x, locref_y = _locref()
    _, ax = viz.visualize_locrefs(_image(), _scmap(), locref_x, locref_y)
    quivers = [c for c in ax.collections if isinstance(c, matplotlib.quiver.Quiver)]
    assert len(quivers) == 1


def test_visualize_locrefs_zoom_width_adjusts_axis_limits():
    locref_x, locref_y = _locref()
    scmap = _scmap()
    zoom_width = 4
    _, ax = viz.visualize_locrefs(_image(), scmap, locref_x, locref_y, zoom_width=zoom_width)

    maxloc = np.unravel_index(np.argmax(scmap), scmap.shape)
    expected_xlim = (maxloc[1] - zoom_width, maxloc[1] + zoom_width)
    assert ax.get_xlim() == pytest.approx(expected_xlim)


def test_visualize_locrefs_no_zoom_does_not_change_axis_limits():
    locref_x, locref_y = _locref()
    _, ax = viz.visualize_locrefs(_image(), _scmap(), locref_x, locref_y, zoom_width=0)
    # x limits should still span the full image width set by form_figure
    xmin, xmax = ax.get_xlim()
    assert xmin == pytest.approx(0)
    assert xmax == pytest.approx(W)


# ---------------------------------------------------------------------------
# visualize_paf
# ---------------------------------------------------------------------------


def _paf(n_fields: int = N_PAF_FIELDS) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(-1, 1, size=(H, W, n_fields, 2)).astype(np.float32)


def test_visualize_paf_returns_figure_and_axes():
    fig, ax = viz.visualize_paf(_image(), _paf())
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_visualize_paf_adds_one_quiver_per_field():
    n_fields = 3
    _, ax = viz.visualize_paf(_image(), _paf(n_fields))
    quivers = [c for c in ax.collections if isinstance(c, matplotlib.quiver.Quiver)]
    assert len(quivers) == n_fields


def test_visualize_paf_defaults_to_red_color():
    n_fields = 2
    _, ax = viz.visualize_paf(_image(), _paf(n_fields))
    quivers = [c for c in ax.collections if isinstance(c, matplotlib.quiver.Quiver)]
    for q in quivers:
        # default color is "r"
        assert q.get_facecolor() is not None


def test_visualize_paf_accepts_custom_colors():
    colors = ["blue", "green"]
    _, ax = viz.visualize_paf(_image(), _paf(len(colors)), colors=colors)
    quivers = [c for c in ax.collections if isinstance(c, matplotlib.quiver.Quiver)]
    assert len(quivers) == len(colors)


def test_visualize_paf_single_field():
    fig, ax = viz.visualize_paf(_image(), _paf(1))
    assert isinstance(fig, plt.Figure)
    quivers = [c for c in ax.collections if isinstance(c, matplotlib.quiver.Quiver)]
    assert len(quivers) == 1
