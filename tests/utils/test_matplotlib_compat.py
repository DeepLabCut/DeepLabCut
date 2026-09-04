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

For the colormap API, covers both the modern code path (exercised against
whatever Matplotlib is actually installed) and the legacy code path, which is
simulated by hiding ``matplotlib.colormaps`` so these tests keep working even
once the real ``matplotlib.cm`` legacy functions are removed upstream.
``remove_artists`` and ``silence_axes_logger`` have no legacy branch; they are
exercised against real ``Axes`` objects and the real logger registry.
"""

from __future__ import annotations

import logging
from unittest import mock

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle

from deeplabcut.core.deprecation import DLCDeprecationWarning
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


@pytest.fixture(autouse=True)
def _cleanup_test_colormap():
    """Ensure the test colormap never leaks into the global registry."""
    yield
    try:
        mc.unregister_colormap(TEST_CMAP_NAME)
    except (ValueError, KeyError):
        pass


def _hide_modern_colormaps_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``hasattr(matplotlib, "colormaps")`` False for the module under test."""
    monkeypatch.delattr(mc.matplotlib, "colormaps", raising=False)


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
# get_colormap_names
# ---------------------------------------------------------------------------


def test_get_colormap_names_contains_known_builtins():
    names = mc.get_colormap_names()
    assert isinstance(names, list)
    assert "viridis" in names


def test_get_colormap_names_legacy_fallback_warns_with_public_name_and_dispatches(monkeypatch):
    _hide_modern_colormaps_api(monkeypatch)
    fake_colormaps = mock.Mock(return_value=["fake_cmap_a", "fake_cmap_b"])
    monkeypatch.setattr(mc.plt, "colormaps", fake_colormaps)

    with pytest.warns(DLCDeprecationWarning, match=r"^get_colormap_names"):
        names = mc.get_colormap_names()

    fake_colormaps.assert_called_once_with()
    assert names == ["fake_cmap_a", "fake_cmap_b"]


# ---------------------------------------------------------------------------
# register_colormap / unregister_colormap (modern API)
# ---------------------------------------------------------------------------


def test_register_and_unregister_colormap_round_trip(custom_cmap):
    mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME)
    assert TEST_CMAP_NAME in mc.get_colormap_names()
    assert isinstance(mc.get_colormap(TEST_CMAP_NAME), Colormap)

    mc.unregister_colormap(TEST_CMAP_NAME)
    assert TEST_CMAP_NAME not in mc.get_colormap_names()


def test_register_colormap_force_overwrites_existing(custom_cmap):
    mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME)

    replacement = ListedColormap(["black", "white"], name=TEST_CMAP_NAME)
    mc.register_colormap(replacement, name=TEST_CMAP_NAME, force=True)

    assert mc.get_colormap(TEST_CMAP_NAME).N == 2


def test_register_colormap_without_force_raises_on_duplicate(custom_cmap):
    mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME)

    with pytest.raises(ValueError):
        mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME, force=False)


# ---------------------------------------------------------------------------
# register_colormap / unregister_colormap (legacy fallback)
# ---------------------------------------------------------------------------
#
# The underlying ``matplotlib.cm.register_cmap``/``unregister_cmap`` functions
# are themselves deprecated and slated for removal, so ``matplotlib.cm`` is
# mocked here: these tests only pin down *our* dispatch logic (which function
# is called, with which arguments, and what warning is raised), not
# Matplotlib's own deprecated behavior.


def test_register_colormap_legacy_fallback_warns_with_public_name_and_dispatches(custom_cmap, monkeypatch):
    _hide_modern_colormaps_api(monkeypatch)
    fake_cm = mock.Mock()
    monkeypatch.setattr(mc.matplotlib, "cm", fake_cm, raising=False)

    with pytest.warns(DLCDeprecationWarning, match=r"^register_colormap"):
        mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME, force=True)

    fake_cm.register_cmap.assert_called_once_with(
        name=TEST_CMAP_NAME,
        cmap=custom_cmap,
        override_builtin=True,
    )


def test_unregister_colormap_legacy_fallback_warns_with_public_name_and_dispatches(monkeypatch):
    _hide_modern_colormaps_api(monkeypatch)
    fake_cm = mock.Mock()
    monkeypatch.setattr(mc.matplotlib, "cm", fake_cm, raising=False)

    with pytest.warns(DLCDeprecationWarning, match=r"^unregister_colormap"):
        mc.unregister_colormap(TEST_CMAP_NAME)

    fake_cm.unregister_cmap.assert_called_once_with(TEST_CMAP_NAME)


def test_register_colormap_legacy_fallback_rejects_duplicate_without_force(custom_cmap, monkeypatch):
    """The legacy path enforces the same duplicate contract as the modern one.

    ``register_cmap(override_builtin=...)`` only governs *builtin* names and
    silently overwrote anything else, so the shim has to raise on its own.
    """
    _hide_modern_colormaps_api(monkeypatch)
    fake_cm = mock.Mock()
    monkeypatch.setattr(mc.matplotlib, "cm", fake_cm, raising=False)
    monkeypatch.setattr(mc.plt, "colormaps", mock.Mock(return_value=[TEST_CMAP_NAME]))

    with pytest.warns(DLCDeprecationWarning), pytest.raises(ValueError, match="already registered"):
        mc.register_colormap(custom_cmap, name=TEST_CMAP_NAME, force=False)

    fake_cm.register_cmap.assert_not_called()


def test_register_colormap_legacy_fallback_falls_back_to_the_cmap_name(custom_cmap, monkeypatch):
    """With ``name`` omitted, the duplicate check uses the colormap's own name."""
    _hide_modern_colormaps_api(monkeypatch)
    monkeypatch.setattr(mc.matplotlib, "cm", mock.Mock(), raising=False)
    monkeypatch.setattr(mc.plt, "colormaps", mock.Mock(return_value=[custom_cmap.name]))

    with pytest.warns(DLCDeprecationWarning), pytest.raises(ValueError, match=custom_cmap.name):
        mc.register_colormap(custom_cmap)


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


def test_remove_artists_on_empty_axes_is_harmless(axes):
    mc.remove_artists(axes)
    assert _counts(axes) == dict.fromkeys(mc.ARTIST_KINDS, 0)


def test_remove_artists_rejects_unknown_kind(axes):
    with pytest.raises(AttributeError):
        mc.remove_artists(axes, "not_a_sublist")


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
