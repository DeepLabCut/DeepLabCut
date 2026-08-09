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

Covers both the modern code path (exercised against whatever Matplotlib is
actually installed) and the legacy code path, which is simulated by hiding
``matplotlib.colormaps`` so these tests keep working even once the real
``matplotlib.cm`` legacy functions are removed upstream.
"""

from __future__ import annotations

from unittest import mock

import pytest
from matplotlib.colors import Colormap, ListedColormap

from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.utils import matplotlib_compat as mc

TEST_CMAP_NAME = "dlc_test_custom_cmap"


@pytest.fixture
def custom_cmap() -> ListedColormap:
    return ListedColormap(["red", "green", "blue"], name=TEST_CMAP_NAME)


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
