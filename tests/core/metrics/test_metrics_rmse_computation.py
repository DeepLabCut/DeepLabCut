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
    computed_results = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    rmse, rmse_cutoff = computed_results["rmse"], computed_results["rmse_pcutoff"]
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

    computed_results = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    rmse, rmse_cutoff = computed_results["rmse"], computed_results["rmse_pcutoff"]
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

    results = compute_rmse(data, False, pcutoff=0.6, oks_bbox_margin=10.0)
    rmse, rmse_cutoff = results["rmse"], results["rmse_pcutoff"]
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, data_unique, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [np.nan, np.nan, 0], [10.0, 10.0, 2]],
            ],
            [  # predicted pose
                [[12.0, 10.0, 0.9], [10.0, 10.0, 0.4], [10.0, 10.0, 0.9]],
            ],
            None, # unique data
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
            None, # unique data
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
            None, # unique data
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
            None, # unique data
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
            None, # unique data
            (0, 0),
        ),
        (  # more detections than GT
            [  # ground truth pose
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
            ],
            [  # predicted pose, no error
                [[10.0, 10.0, 2], [20.0, 20.0, 2], [30.0, 30.0, 2]],
                [[40.0, 40.0, 2], [50.0, 50.0, 2], [60.0, 60.0, 2]],
                [[70.0, 70.0, 2], [80.0, 80.0, 2], [90.0, 90.0, 2]],
            ],
            None, # unique data
            (0, 0),
        ),
        (
                [  # ground truth pose
                    [[10.0, 10.0, 2], [np.nan, np.nan, 0], [10.0, 10.0, 2]],
                ],
                [  # predicted pose
                    [[12.0, 10.0, 0.9], [10.0, 10.0, 0.4], [10.0, 10.0, 0.9]],
                ],
                (  # unique data
                        [[[20, 20, 2], [22, 23, 2]]],
                        [[[20, 20, 0.8], [22, 23, 0.7]]]
                ),
                (0.5, 0.5),  # error 2 on one, 0 on the other; only 2 valid GT
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
                (  # unique data
                        [], # missing ground truth for unique bodyparts
                        [[[20, 20, 0.8], [22, 23, 0.7]]]
                ),
                (0, 0),  # all pose perfect
        ),
    ],
)
def test_detection_rmse(gt: list, pred: list, data_unique:tuple[list, list]|None, result: tuple[float, float]):
    data = [(np.asarray(gt), np.asarray(pred))]
    data_unique = [(np.asarray(data_unique[0]), np.asarray(data_unique[1]))] if data_unique else None
    expected_rmse, expected_rmse_cutoff = result
    rmse, rmse_cutoff = compute_detection_rmse(data, pcutoff=0.6, data_unique=data_unique)
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, unique_gt, unique_pred, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
                [[20.0, 24.0, 0.2], [20.0, 24.0, 0.2], [20.0, 20.0, 0.2]],
            ],
            [  # Unique GT
                [[10.0, 10.0, 2], [10.0, 10.0, 2]],
            ],
            [  # Unique Pred
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
            ],
            # 4 pixel error on 2 keypoints, 0 error on 5 keypoints
            (1.0, 0.0),
        ),
        (
            [np.zeros((0, 3, 2))],  # no GT pose
            [  # predicted pose
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
            ],
            [  # Unique GT
                [[10.0, 10.0, 2], [10.0, 10.0, 2]],
            ],
            [  # Unique Pred
                [[15.0, 10.0, 0.5], [11.0, 10.0, 0.9]],
            ],
            # 5 pixel error on 1 keypoint, 1 pixel error on the other
            (3.0, 1.0),
        ),
    ],
)
def test_rmse_with_unique(
    gt: list,
    pred: list,
    unique_gt: list,
    unique_pred: list,
    result: tuple[float, float]
) -> None:
    data = [(np.asarray(gt), np.asarray(pred))]
    data_unique = [(np.asarray(unique_gt), np.asarray(unique_pred))]
    expected_rmse, expected_rmse_cutoff = result

    results = compute_rmse(
        data, False, pcutoff=0.6, data_unique=data_unique, oks_bbox_margin=10.0,
    )
    rmse, rmse_cutoff = results["rmse"], results["rmse_pcutoff"]
    assert_almost_equal(rmse, expected_rmse)
    assert_almost_equal(rmse_cutoff, expected_rmse_cutoff)


@pytest.mark.parametrize(
    "gt, pred, unique_gt, unique_pred, result",
    [
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
                [[20.0, 24.0, 0.2], [20.0, 24.0, 0.2], [20.0, 20.0, 0.2]],
            ],
            [  # Unique GT
                [[10.0, 10.0, 2], [10.0, 10.0, 2]],
            ],
            [  # Unique Pred
                [[10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
            ],
            # 4 pixel error on 2 keypoints, 0 error on 5 keypoints
            [
                (1.0, 0.0),
                [2.0, 2.0, 0.0],
                [0.0, 0.0]
            ],
        ),
        (
            [  # ground truth pose
                [[10.0, 10.0, 2], [10.0, 10.0, 2], [10.0, 10.0, 2]],
                [[20.0, 20.0, 2], [20.0, 20.0, 2], [20.0, 20.0, 2]],
            ],
            [  # predicted pose
                [[10.0, 12.0, 0.9], [10.0, 10.0, 0.9], [10.0, 10.0, 0.9]],
                [[20.0, 24.0, 0.7], [20.0, 24.0, 0.6], [20.0, 20.0, 0.8]],
            ],
            [  # Unique GT
                [[10.0, 10.0, 2], [10.0, 10.0, 2]],
            ],
            [  # Unique Pred
                [[12.0, 10.0, 0.9], [11.0, 10.0, 0.9]],
            ],
            [  # errors: 3 with 0px, 1 with 1px, 2 with 2px, 2 with 4px => 13/8
                (1.625, 1.625),
                [3.0, 2.0, 0.0],
                [2.0, 1.0]
            ],
        ),
    ],
)
def test_rmse_per_bodypart_with_unique(
    gt: list,
    pred: list,
    unique_gt: list,
    unique_pred: list,
    result: tuple[tuple[float, float], list[float], list[float]]
) -> None:
    data = [(np.asarray(gt), np.asarray(pred))]
    data_unique = [(np.asarray(unique_gt), np.asarray(unique_pred))]
    expected_rmse, expected_rmse_cutoff = result[0]
    bodypart_rmse = result[1]
    unique_rmse = result[2]

    results = compute_rmse(
        data,
        single_animal=False,
        pcutoff=0.6,
        data_unique=data_unique,
        per_keypoint_results=True,
        oks_bbox_margin=10.0,
    )
    assert_almost_equal(results["rmse"], expected_rmse)
    assert_almost_equal(results["rmse_pcutoff"], expected_rmse_cutoff)
    for bpt_index, bpt_rmse in enumerate(bodypart_rmse):
        key = f"rmse_keypoint_{bpt_index}"
        assert key in results
        assert_almost_equal(results[key], bpt_rmse)

    for bpt_index, bpt_rmse in enumerate(unique_rmse):
        key = f"rmse_unique_keypoint_{bpt_index}"
        assert key in results
        assert_almost_equal(results[key], bpt_rmse)
