#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import numpy as np
import pandas as pd
import pytest
from deeplabcut.pose_estimation_3d import triangulation


@pytest.fixture(scope="session")
def stereo_params():
    params = dict()
    for i in range(1, 3):
        params[f"cameraMatrix{i}"] = np.random.rand(3, 3)
        params[f"distCoeffs{i}"] = np.random.rand(1, 5)
        params[f"P{i}"] = np.random.rand(3, 4)
        params[f"R{i}"] = np.eye(3)
    return params


def test_undistort_points(stereo_params):
    points = np.random.rand(100, 20 * 3)
    points_undistorted = triangulation._undistort_points(
        points,
        stereo_params["cameraMatrix1"],
        stereo_params["distCoeffs1"],
        stereo_params["P1"],
        stereo_params["R1"],
    )
    # Test that shape was preserved after vectorization
    assert np.shape(points_undistorted) == np.shape(points)


@pytest.mark.parametrize(
    "n_view_pairs, is_multi",
    [(i, flag) for i in range(1, 7, 2) for flag in (False, True)],
)
def test_undistort_views(n_view_pairs, is_multi, stereo_params):
    df = pd.read_hdf("tests/data/montblanc_tracks.h5")
    if not is_multi:
        df = df.xs("bird1", level="individuals", axis=1)

    view_pairs = [(df, df) for _ in range(n_view_pairs)]
    cam_params = {
        f"camera-1-camera-{i}": stereo_params for i in range(2, n_view_pairs + 2)
    }
    dfs = triangulation._undistort_views(view_pairs, cam_params)
    assert len(dfs) == n_view_pairs
    assert all(len(pair) == 2 for pair in dfs)
    assert len(dfs[0][0].columns.levels) == (4 if is_multi else 3)
