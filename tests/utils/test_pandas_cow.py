#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

"""Verify that ``configure_pandas_future_if_enabled`` (copy_on_write=True)
surfaces mutation patterns that should be guarded by ``copy=True``.
"""

# NOTE: @C-Achard 2026-09-04 : [CLEANUP-COLLECTEDDATA-FIXTURES] _make_dlc_df below is another variant of the
# CollectedData frame that at least six test modules each build for themselves, this one without a row
# index. Reuse target: the collected_data fixture in tests/utils/test_skeleton.py, once it
# is promoted to a shared conftest.
import numpy as np
import pandas as pd
import pytest

from deeplabcut.utils.pandas_future_mode import (
    configure_pandas_future_if_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_cow_true(monkeypatch) -> None:
    """Configure pandas CoW via the DLC future-mode machinery."""
    monkeypatch.setattr(pd.options.future, "infer_string", pd.options.future.infer_string)
    monkeypatch.setattr(pd.options.mode, "copy_on_write", pd.options.mode.copy_on_write)
    monkeypatch.setenv("DLC_PANDAS_FUTURE", "1")
    configure_pandas_future_if_enabled()


def _make_dlc_df(n_frames: int = 5) -> pd.DataFrame:
    """Build a small DLC-style MultiIndex DataFrame (scorer x bodypart x coord)."""
    bodyparts = ["snout", "leftear", "rightear"]
    coords = ["x", "y", "likelihood"]
    arrays = {}
    for bp in bodyparts:
        for c in coords:
            arrays[("scorer1", bp, c)] = np.random.randn(n_frames)
    df = pd.DataFrame(arrays)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns,
        names=["scorer", "bodyparts", "coords"],
    )
    return df


# ---------------------------------------------------------------------------
# Sanity: the machinery actually engages.
# ---------------------------------------------------------------------------


def test_future_mode_sets_cow_true(monkeypatch):
    """``configure_pandas_future_if_enabled()`` sets copy_on_write=True."""
    _configure_cow_true(monkeypatch)
    assert pd.options.mode.copy_on_write is True, f"Expected copy_on_write=True, got {pd.options.mode.copy_on_write!r}"
    assert pd.options.future.infer_string is True


# ---------------------------------------------------------------------------
# The PR's patterns: extracting a numpy array, then mutating it in place.
# Without copy=True each would blow up at runtime under pandas 3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extract",
    [
        pytest.param(
            lambda df: df.to_numpy().reshape((len(df), -1, 3)),
            id="to_numpy_reshape",
        ),
        pytest.param(
            lambda df: df.values.reshape((len(df), -1, 3)),
            id="values_reshape",
        ),
        pytest.param(
            lambda df: df.loc[:, df.columns.get_level_values("coords").isin(("x", "y"))].to_numpy(),
            id="column_subset",
        ),
        pytest.param(
            lambda df: df.loc[df.index[0]].to_numpy().reshape(-1, 3),
            id="single_row",
        ),
    ],
)
def test_future_mode_makes_numpy_arrays_read_only(extract, monkeypatch):
    """``configure_pandas_future_if_enabled()`` with copy_on_write=True
    returns read-only numpy arrays that cannot be mutated in place.

    These are the patterns that this PR guards with ``copy=True``.
    Without the guard, each would raise ``ValueError`` (read-only array)
    under pandas 3.
    """
    _configure_cow_true(monkeypatch)
    df = _make_dlc_df()

    arr = extract(df)

    assert not arr.flags.writeable, "array unexpectedly writeable — CoW may not be active"

    with pytest.raises(ValueError, match="read-only"):
        arr.flat[0] = 999.0
