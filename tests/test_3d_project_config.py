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
"""Regression tests: 3D entry points must read a 3D config.yaml without 2D validation.

See https://github.com/DeepLabCut/DeepLabCut/issues/3585. Each entry point is run on a
freshly created 3D project; it must get past reading the config and fail later, on
the data that a fresh project does not have yet.
"""

import pytest

from deeplabcut.create_project.new_3d import create_new_project_3d
from deeplabcut.pose_estimation_3d import camera_calibration, plotting3D, triangulation
from deeplabcut.utils import auxiliaryfunctions


@pytest.fixture
def config_3d(tmp_path):
    return create_new_project_3d("trial", "tester", num_cameras=2, working_directory=str(tmp_path))


@pytest.mark.parametrize(
    "run, error, match",
    [
        (
            lambda cfg, tmp: camera_calibration.calibrate_cameras(cfg, cbrow=8, cbcol=6),
            Exception,
            "No calibration images found",
        ),
        (
            lambda cfg, tmp: camera_calibration.check_undistortion(cfg, cbrow=8, cbcol=6),
            FileNotFoundError,
            "stereo_params.pickle",
        ),
        (
            lambda cfg, tmp: triangulation.triangulate(cfg, str(tmp)),
            Exception,
            "config_file_camera-1 does not exist",
        ),
        (
            lambda cfg, tmp: triangulation.undistort_points(cfg, [tmp / "a.h5", tmp / "b.h5"], None),
            FileNotFoundError,
            "Dataframe path",
        ),
        (
            lambda cfg, tmp: plotting3D.create_labeled_video_3d(cfg, [str(tmp)]),
            Exception,
            "No corresponding video file",
        ),
    ],
    ids=["calibrate_cameras", "check_undistortion", "triangulate", "undistort_points", "create_labeled_video_3d"],
)
def test_3d_entry_points_read_3d_config(config_3d, tmp_path, run, error, match):
    with pytest.raises(error, match=match):
        run(config_3d, tmp_path)


def test_calibrate_cameras_keeps_3d_config_keys(config_3d):
    before = auxiliaryfunctions.read_plainconfig(config_3d)

    with pytest.raises(Exception, match="No calibration images found"):
        camera_calibration.calibrate_cameras(config_3d, cbrow=8, cbcol=6)

    assert auxiliaryfunctions.read_plainconfig(config_3d) == before
