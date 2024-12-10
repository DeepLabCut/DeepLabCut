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
"""General tests for the metrics API"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import deeplabcut.core.metrics as metrics


def _get_gt_and_pred_with_constant_err(
    num_idv: int, num_bpt: int, error: float
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.arange(num_idv * num_bpt * 3).astype(float).reshape((num_idv, num_bpt, 3))
    gt[..., 2] = 2
    predictions = gt.copy()
    predictions[..., 2] = 0.9
    predictions[..., :2] += error
    return gt, predictions


def test_computing_metrics_with_no_predictions():
    gt = np.arange(5 * 6 * 3).astype(float).reshape((5, 6, 3))
    gt[..., 2] = 2
    metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": np.zeros((0, 12, 3))},
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_computing_metrics_with_constant_error(error):
    # only works for small errors: otherwise another matching can be found
    gt, predictions = _get_gt_and_pred_with_constant_err(5, 6, error)
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_metrics_with_unique_with_constant_error(error):
    # only works for small errors: otherwise another matching can be found
    gt, predictions = _get_gt_and_pred_with_constant_err(5, 6, error)
    gt_unique, pred_unique = _get_gt_and_pred_with_constant_err(1, 8, error)
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        unique_bodypart_gt={"image": gt_unique},
        unique_bodypart_poses={"image": pred_unique},
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_metrics_per_bpt_with_unique_with_constant_error(error):
    # only works for small errors: otherwise another matching can be found
    gt, predictions = _get_gt_and_pred_with_constant_err(5, 6, error)
    gt_unique, pred_unique = _get_gt_and_pred_with_constant_err(1, 8, error)
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        unique_bodypart_gt={"image": gt_unique},
        unique_bodypart_poses={"image": pred_unique},
        per_keypoint_rmse=True,
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)

    for bpt_idx in range(gt.shape[1]):
        key = f"rmse_keypoint_{bpt_idx}"
        assert key in results
        assert_almost_equal(results[key], np.sqrt(2) * error)
    for bpt_idx in range(gt_unique.shape[1]):
        key = f"rmse_unique_keypoint_{bpt_idx}"
        assert key in results
        assert_almost_equal(results[key], np.sqrt(2) * error)


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_computing_metrics_single_animal(error):
    # only works for small errors: otherwise another matching can be found
    gt = np.arange(6 * 3).astype(float).reshape((1, 6, 3))
    gt[..., 2] = 2
    predictions = gt.copy()
    predictions[..., 2] = 0.9
    predictions[..., :2] += error
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        single_animal=True,
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)

