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
"""Tests for ``deeplabcut.utils.matplotlib_compat``.

Every helper is exercised against the installed Matplotlib,
real ``Axes`` objects and the real logger registry.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle

from deeplabcut.utils import matplotlib_compat as mc

TEST_CMAP_NAME = "dlc_test_custom_cmap"


@pytest.fixture
def custom_cmap() -> ListedColormap:
    return ListedColormap(["red", "green", "blue"], name=TEST_CMAP_NAME)


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ---------------------------------------------------------------------------
# get_colormap
# ---------------------------------------------------------------------------


def test_get_colormap_returns_colormap_instance():
    cmap = mc.get_colormap("viridis")
    assert isinstance(cmap, Colormap)
    assert cmap.name == "viridis"


def test_get_colormap_resamples_to_lut():
    cmap = mc.get_colormap("viridis", 5)
    assert cmap.N == 5


def test_get_colormap_accepts_none():
    cmap = mc.get_colormap(None)
    assert isinstance(cmap, Colormap)


def test_get_colormap_passes_through_colormap_instance(custom_cmap):
    result = mc.get_colormap(custom_cmap)
    assert result.name == custom_cmap.name


# ---------------------------------------------------------------------------
# remove_artists
# ---------------------------------------------------------------------------


def _populate(ax) -> None:
    """Add exactly one artist of every kind in ``ARTIST_KINDS``.

    ``AnchoredText`` rather than a second ``Rectangle`` for the ``artists``
    sublist: that sublist excludes ``Patch``, so an ``add_artist(Rectangle)``
    would land in ``patches`` and leave ``artists`` empty.
    """
    ax.plot([0, 1], [0, 1])
    ax.scatter([0, 1], [1, 0])
    ax.add_patch(Rectangle((0, 0), 1, 1))
    ax.imshow([[0, 1], [1, 0]])
    ax.add_artist(AnchoredText("txt", loc="upper left"))


def _counts(ax) -> dict[str, int]:
    return {kind: len(getattr(ax, kind)) for kind in mc.ARTIST_KINDS}


def test_artist_kinds_are_real_axes_attributes(axes):
    """Pin the stringly-typed kind names against the Axes API."""
    for kind in mc.ARTIST_KINDS:
        assert hasattr(axes, kind), f"Axes has no {kind!r} sublist"


def test_remove_artists_clears_every_default_kind(axes):
    _populate(axes)
    # Every kind must actually be populated, or the assertion below is vacuous.
    assert all(count > 0 for count in _counts(axes).values())

    mc.remove_artists(axes)

    assert _counts(axes) == dict.fromkeys(mc.ARTIST_KINDS, 0)


def test_remove_artists_only_clears_requested_kinds(axes):
    _populate(axes)
    before = _counts(axes)

    mc.remove_artists(axes, "collections")

    after = _counts(axes)
    assert after["collections"] == 0
    for kind in mc.ARTIST_KINDS:
        if kind != "collections":
            assert after[kind] == before[kind]


def test_remove_artists_removes_every_artist_of_a_kind(axes):
    """Regression guard: iterating a live ArtistList while removing skips artists."""
    for _ in range(5):
        axes.scatter([0, 1], [1, 0])
    assert len(axes.collections) == 5

    mc.remove_artists(axes, "collections")

    assert len(axes.collections) == 0


# ---------------------------------------------------------------------------
# silence_axes_logger
# ---------------------------------------------------------------------------


@pytest.fixture
def axes_logger():
    """Yield Matplotlib's Axes logger, restoring its level afterwards."""
    logger = logging.getLogger(mc.AXES_LOGGER_NAME)
    original = logger.level
    yield logger
    logger.setLevel(original)


def test_axes_logger_name_matches_matplotlibs_own_logger():
    """``AXES_LOGGER_NAME`` must resolve to the logger Matplotlib actually uses.

    ``silence_axes_logger`` reaches that logger by name instead of importing
    ``_log``, so a module rename upstream would silently stop suppressing
    messages rather than raising.  This is the assertion that turns such a
    rename back into a loud failure, and the one place where importing the
    private module is the point.
    """
    from matplotlib.axes._axes import _log

    assert logging.getLogger(mc.AXES_LOGGER_NAME) is _log


def test_silence_axes_logger_sets_the_level(axes_logger):
    axes_logger.setLevel(logging.DEBUG)

    mc.silence_axes_logger()

    assert axes_logger.level == logging.ERROR


def test_silence_axes_logger_accepts_an_explicit_level(axes_logger):
    mc.silence_axes_logger(logging.CRITICAL)

    assert axes_logger.level == logging.CRITICAL
