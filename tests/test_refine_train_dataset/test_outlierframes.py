from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deeplabcut.refine_training_dataset import outlier_frames

# ----------------------------
# Helpers / fixtures
# ----------------------------

STATS = [
    "distance",
    "sig",
    "meanx",
    "meany",
    "lowerCIx",
    "higherCIx",
    "lowerCIy",
    "higherCIy",
]


@pytest.fixture
def patch_hdf_write(monkeypatch):
    """
    Avoid filesystem / pytables dependency when storeoutput='full' is used.
    Also lets us assert that the write path is still exercised.
    """
    mock = MagicMock()
    monkeypatch.setattr(pd.DataFrame, "to_hdf", mock)
    return mock


@pytest.fixture
def patch_fit_sarimax(monkeypatch):
    def fake_fit_sarimax_model(x, p, p_bound, alpha, ARdegree, MAdegree):
        x = np.asarray(x, dtype=float)
        mean = x.copy()
        ci = np.c_[mean - 1.0, mean + 1.0]
        return mean, ci

    mock = MagicMock(side_effect=fake_fit_sarimax_model)
    monkeypatch.setattr(outlier_frames, "FitSARIMAXModel", mock)
    return mock


@pytest.fixture
def sparse_multianimal_df():
    """
    maDLC-like sparse layout:
      - 2 individuals with shared bodyparts
      - unique bodyparts present only under a special 'single' bucket
    This breaks if reconstructed with the full Cartesian product of the non-'coords' levels,
    e.g. multi-animal projects with unique bodyparts were previously
    producing many extra columns for the non-existent combinations of individual x unique bodypart.
    """
    n_frames = 7
    scorer = "DLC_scorer"
    individuals = ["ind1", "ind2"]
    shared_bodyparts = [f"shared_{i}" for i in range(18)]
    unique_bodyparts = [f"unique_{i}" for i in range(4)]
    coords = ["x", "y", "likelihood"]

    tuples = []

    # Shared bodyparts for each real individual
    for ind in individuals:
        for bp in shared_bodyparts:
            for c in coords:
                tuples.append((scorer, ind, bp, c))

    # Unique bodyparts only under a special bucket
    for bp in unique_bodyparts:
        for c in coords:
            tuples.append((scorer, "single", bp, c))

    columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "individuals", "bodyparts", "coords"])

    # 18 shared * 2 + 4 unique = 40 streams, each with x/y/likelihood
    assert len(columns) == 40 * 3

    rng = np.random.default_rng(42)
    values = rng.normal(size=(n_frames, len(columns)))

    # Keep likelihood valid / boring
    likelihood_mask = columns.get_level_values("coords") == "likelihood"
    values[:, likelihood_mask] = 0.9

    df = pd.DataFrame(values, columns=columns)
    return df


@pytest.fixture
def dense_multianimal_df():
    """
    Dense/full-combination layout:
      every individual x bodypart combination exists.
    For this topology, the old from_product(...) logic and the new "preserve
    actual tuples" logic should produce the same output columns (assuming the
    dataframe is created in canonical product order, which we do here).
    """
    n_frames = 5
    scorer = "DLC_scorer"
    individuals = ["ind1", "ind2"]
    bodyparts = ["nose", "tail", "paw"]
    coords = ["x", "y", "likelihood"]

    tuples = [(scorer, ind, bp, c) for ind in individuals for bp in bodyparts for c in coords]

    columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "individuals", "bodyparts", "coords"])

    rng = np.random.default_rng(42)
    values = rng.normal(size=(n_frames, len(columns)))
    likelihood_mask = columns.get_level_values("coords") == "likelihood"
    values[:, likelihood_mask] = 0.95

    df = pd.DataFrame(values, columns=columns)
    return df


def _expected_output_columns_from_actual_streams(df):
    """
    Expected output columns preserve actual non-'coords' tuples and append the 8 derived stats.
    """
    base_cols = df.xs("x", axis=1, level="coords", drop_level=True).columns
    return pd.MultiIndex.from_tuples(
        [(tuple(col) if isinstance(col, tuple) else (col,)) + (stat,) for col in base_cols for stat in STATS],
        names=df.columns.names,
    )


def _expected_output_columns_from_dense_product(df):
    """
    Expected output columns for the previous implementation:
    full Cartesian product of all non-'coords' levels, then the 8 derived stats.
    This is only correct / behavior-preserving for dense layouts.
    """
    columns = df.columns
    prod = []
    for i in range(columns.nlevels - 1):
        prod.append(columns.get_level_values(i).unique())
    prod.append(STATS)
    return pd.MultiIndex.from_product(prod, names=columns.names)


# ----------------------------
# Tests
# ----------------------------


def test_compute_deviations_regression_sparse_unique_bodyparts(
    sparse_multianimal_df,
    patch_fit_sarimax,
    patch_hdf_write,
):
    """
    Regression test for the following maDLC unique-bodypart bug:
    output columns must match the actual sparse stream layout rather than an
    inflated Cartesian product of all non-'coords' level values.
    """
    df = sparse_multianimal_df
    n_frames = len(df)

    d, o, data = outlier_frames.compute_deviations(
        df,
        dataname="dummy.h5",
        p_bound=0.01,
        alpha=0.01,
        ARdegree=3,
        MAdegree=1,
        storeoutput="full",
    )

    # There are 40 real streams in the sparse fixture
    n_streams = 40

    # Shape sanity checks
    assert d.shape == (n_frames,)
    assert o.shape == (n_frames,)
    assert data.shape == (n_frames, n_streams * 8)

    # Column layout must preserve only the actual streams
    expected_columns = _expected_output_columns_from_actual_streams(df)
    assert data.columns.equals(expected_columns)

    # xs(...) on the last level should still work exactly as before
    distance = data.xs("distance", axis=1, level=-1)
    sig = data.xs("sig", axis=1, level=-1)
    assert distance.shape == (n_frames, n_streams)
    assert sig.shape == (n_frames, n_streams)

    # With the fake fitter, predictions equal observations => zero distances and sig
    np.testing.assert_allclose(d, 0.0)
    np.testing.assert_allclose(o, 0.0)

    # FitSARIMAXModel should be called twice per stream (x and y)
    assert patch_fit_sarimax.call_count == 2 * n_streams

    # "full" path should still try to persist the result
    patch_hdf_write.assert_called_once()


def test_compute_deviations_behavior_preserved_for_dense_layout(
    dense_multianimal_df,
    patch_fit_sarimax,
    patch_hdf_write,
):
    """
    Behavior-preserved check:
    for a dense layout where every combination exists, the fixed implementation
    should produce the same columns that the old from_product(...) logic would
    have produced.
    """
    df = dense_multianimal_df
    n_frames = len(df)
    n_streams = len(df.xs("x", axis=1, level="coords", drop_level=True).columns)

    d, o, data = outlier_frames.compute_deviations(
        df,
        dataname="dummy.h5",
        p_bound=0.01,
        alpha=0.01,
        ARdegree=3,
        MAdegree=1,
        storeoutput="full",
    )

    # Basic shape / output checks
    assert d.shape == (n_frames,)
    assert o.shape == (n_frames,)
    assert data.shape == (n_frames, n_streams * 8)

    # For dense data, new behavior should match old dense-product behavior exactly
    expected_old_dense_columns = _expected_output_columns_from_dense_product(df)
    expected_new_columns = _expected_output_columns_from_actual_streams(df)

    # Sanity check of the fixture assumption:
    # in the dense case, these should indeed be identical.
    assert expected_new_columns.equals(expected_old_dense_columns)

    # Actual output should match that shared expected index
    assert data.columns.equals(expected_old_dense_columns)

    # Still selectable by derived-stat level
    assert data.xs("distance", axis=1, level=-1).shape == (n_frames, n_streams)
    assert data.xs("sig", axis=1, level=-1).shape == (n_frames, n_streams)

    # Deterministic fake fitter
    np.testing.assert_allclose(d, 0.0)
    np.testing.assert_allclose(o, 0.0)

    assert patch_fit_sarimax.call_count == 2 * n_streams
    patch_hdf_write.assert_called_once()
