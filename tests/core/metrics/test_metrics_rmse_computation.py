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
"""Tests RMSE computation"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from deeplabcut.core.metrics.distance_metrics import (
    compute_detection_rmse,
    compute_rmse,
)


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (
            [  # ground truth pose
                [[100.0, 10.0, 2], [150.0, 15.0, 2], [200.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[100.0, 10.0, 0.9], [150.0, 15.0, 0.8], [200.0, 20.0, 0.8]],
            ],
            (0, 0),
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[12.0, 10.0, 0.9], [12.0, 10.0, 0.9], [12.0, 10.0, 0.9]],
                [[22.0, 20.0, 0.9], [22.0, 20.0, 0.9], [22.0, 20.0, 0.9]],
            ],
            (2, 2),
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[10.0, 12.0, 0.9], [10.0, 12.0, 0.9], [10.0, 12.0, 0.9]],
                [[20.0, 22.0, 0.9], [20.0, 22.0, 0.9], [20.0, 22.0, 0.9]],
            ],
            (2, 2),
        ),
    ],
)
def test_rmse_single_image(gt: list, pred: list, result: tuple[float, float]):
    data = [(np.asarray(gt), np.asarray(pred))]
    rmse, rmse_cutoff = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    expected_rmse, expected_rmse_cutoff = result
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
                [[20.0, 22.0, 0.2], [20.0, 22.0, 0.2], [20.0, 22.0, 0.2]],
            ],
            (1, 0),  # 2 pixel error on half of keypoints, 0 on the other half
        ),
    ],
)
def test_rmse_pcutoff(gt: list, pred: list, result: tuple[float, float]):
    data = [(np.asarray(gt), np.asarray(pred))]
    expected_rmse, expected_rmse_cutoff = result

    rmse, rmse_cutoff = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [float("nan"), float("nan"), 0], [10.0, 10.0, 2]],
            ],
            [  # predicted pose
                [[12.0, 10.0, 0.9], [10.0, 10.0, 0.4], [10.0, 10.0, 0.9]],
            ],
            (1, 1),  # only 2 valid ground truth bodyparts
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [float("nan"), float("nan"), 0]],
                [[float("nan"), float("nan"), 0], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose, swapped prediction order
                [[20.0, 20.0, 0.9], [21.0, 20.0, 0.9], [21.0, 20.0, 0.9]],
                [[15.0, 10.0, 0.4], [15.0, 10.0, 0.4], [10.0, 10.0, 0.9]],
            ],
            (3, 1),  # only 2 valid GT bodyparts
        ),
    ],
)
def test_rmse_with_nans(gt: list, pred: list, result: tuple[float, float]):
    data = [(np.asarray(gt), np.asarray(pred))]
    expected_rmse, expected_rmse_cutoff = result

    rmse, rmse_cutoff = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [np.nan, np.nan, 0], [10.0, 10.0, 2]],
            ],
            [  # predicted pose
                [[12.0, 10.0, 0.9], [10.0, 10.0, 0.4], [10.0, 10.0, 0.9]],
            ],
            (1, 1),  # error 2 on one, 0 on the other; only 2 valid GT
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            [  # predicted pose, perfect detections but mis-assembled
                [[10.0, 10.0, 0.9], [50.0, 50.0, 0.9], [30.0, 30.0, 0.9]],
                [[40.0, 40.0, 0.9], [20.0, 20.0, 0.4], [60.0, 60.0, 0.9]],
            ],
            (0, 0),  # all pose perfect
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            [  # predicted pose, small error in pose and mis-assembled
                [[12.0, 10.0, 0.9], [52.0, 50.0, 0.9], [32.0, 30.0, 0.9]],
                [[42.0, 40.0, 0.9], [18.0, 20.0, 0.4], [62.0, 60.0, 0.9]],
            ],
            (2, 2),  # pixel error of 2 on x-axis for all predictions
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            [  # predicted pose, small error in low-conf pose and mis-assembled
                [[12.0, 10.0, 0.4], [50.0, 50.0, 0.9], [30.0, 30.0, 0.9]],
                [[40.0, 40.0, 0.9], [22.0, 20.0, 0.4], [62.0, 60.0, 0.4]],
            ],
            (1, 0),  # error of 2 on half, 0 on the other half (with good conf)
        ),
        (  # more ground truth than detections
            [  # ground truth pose
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
            ],
            [  # predicted pose, no error
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            (0, 0),
        ),
        (  # more detections than GT
            [  # predicted pose, no error
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            [  # ground truth pose
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
            ],
            (0, 0),
        ),
    ],
)
def test_detection_rmse(gt: list, pred: list, result: tuple[float, float]):
    data = [(np.asarray(gt), np.asarray(pred))]
    expected_rmse, expected_rmse_cutoff = result
    rmse, rmse_cutoff = compute_detection_rmse(data, pcutoff=0.6)
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)
