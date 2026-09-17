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
"""General tests for the metrics API."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import deeplabcut.core.metrics as metrics


def _get_gt_and_pred_with_constant_err(num_idv: int, num_bpt: int, error: float) -> tuple[np.ndarray, np.ndarray]:
    gt = np.arange(num_idv * num_bpt * 3).astype(float).reshape((num_idv, num_bpt, 3))
    gt[..., 2] = 2
    predictions = gt.copy()
    predictions[..., 2] = 0.9
    predictions[..., :2] += error
    return gt, predictions


def test_computing_metrics_with_no_predictions():
    gt = np.arange(5 * 6 * 3).astype(float).reshape((5, 6, 3))
    gt[..., 2] = 2
    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": np.zeros((0, 12, 3))},
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    # ground truth exists but nothing was predicted -> score of 0
    # != the NaN case (OKS cannot be computed at all)
    assert results["mAP"] == 0
    assert results["mAR"] == 0


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


@pytest.mark.parametrize("error", [0.5, 1, 2])
def test_computing_metrics_single_animal_single_keypoint(error):
    # see https://github.com/DeepLabCut/DeepLabCut/issues/3518: RMSE must be finite
    # for one-bodypart projects, and mAP/mAR undefined rather than 0
    gt = np.arange(3 * 1 * 1 * 3).astype(float).reshape((3, 1, 1, 3))
    gt[..., 2] = 2
    predictions = gt.copy()
    predictions[..., 2] = 0.9
    predictions[..., :2] += error

    results = metrics.compute_metrics(
        ground_truth={f"image{i}": img_gt for i, img_gt in enumerate(gt)},
        predictions={f"image{i}": img_pred for i, img_pred in enumerate(predictions)},
        single_animal=True,
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["rmse"], np.sqrt(2) * error)
    assert_almost_equal(results["rmse_pcutoff"], np.sqrt(2) * error)
    assert np.isnan(results["mAP"])
    assert np.isnan(results["mAR"])


@pytest.mark.parametrize(
    "gt_xy, description",
    [
        pytest.param([[10.0, 10.0], [30.0, 10.0]], "horizontally aligned", id="horizontal"),
        pytest.param([[10.0, 10.0], [10.0, 30.0]], "vertically aligned", id="vertical"),
        pytest.param([[10.0, 10.0], [10.0, 10.0]], "coincident", id="coincident"),
    ],
)
def test_metrics_degenerate_gt_pose_gives_undefined_map(gt_xy, description):
    """A GT pose with no spatial extent cannot be scored by OKS.

    Such a pose passes the ">= 2 visible keypoints" filter, but OKS normalizes by the
    area the pose covers, so `calc_object_keypoint_similarity` returns NaN for it. No
    prediction can ever match it, and reporting mAP/mAR of 0 would suggest the model
    scored badly rather than that the metric is not computable.
    """
    gt = np.array([[[x, y, 2.0] for x, y in gt_xy]])
    predictions = gt.copy()
    predictions[..., 2] = 0.9

    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        single_animal=False,
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert np.isnan(results["mAP"]), f"{description} GT reported a score of {results['mAP']}"
    assert np.isnan(results["mAR"])


def test_metrics_degenerate_gt_pose_is_scored_when_margin_gives_it_an_area():
    """`oks_bbox_margin` pads the pose, so a degenerate pose becomes scoreable."""
    gt = np.array([[[10.0, 10.0, 2.0], [30.0, 10.0, 2.0]]])
    predictions = gt.copy()
    predictions[..., 2] = 0.9

    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        single_animal=False,
        oks_bbox_margin=5,
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["mAP"], 100)
    assert_almost_equal(results["mAR"], 100)


def test_metrics_degenerate_gt_pose_does_not_penalise_scoreable_poses():
    """A pose OKS cannot score must not count against recall for the others."""
    scoreable = [[10.0, 10.0, 2.0], [30.0, 40.0, 2.0]]
    degenerate = [[100.0, 100.0, 2.0], [140.0, 100.0, 2.0]]

    gt = np.array([scoreable, degenerate])
    predictions = np.array([scoreable])  # only the scoreable pose is predicted
    predictions[..., 2] = 0.9

    results = metrics.compute_metrics(
        ground_truth={"image": gt},
        predictions={"image": predictions},
        single_animal=False,
        unique_bodypart_gt=None,
        unique_bodypart_poses=None,
    )
    assert_almost_equal(results["mAP"], 100)
    assert_almost_equal(results["mAR"], 100)
