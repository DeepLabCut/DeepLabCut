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
"""API methods to get metrics for deep learning models"""
from __future__ import annotations

import numpy as np

import deeplabcut.core.metrics.distance_metrics as distance_metrics


def compute_metrics(
    ground_truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    unique_bodypart_gt: dict[str, np.ndarray] | None = None,
    unique_bodypart_poses: dict[str, np.ndarray] | None = None,
    pcutoff: float = -1,
    oks_bbox_margin: int = 0,
    oks_sigma: float = 0.1,
) -> dict:
    """Computes pose estimation performance metrics

    Given ground truth pose labels and predictions on a dataset, computes RMSE and pose
    mAP/mAR using OKS.

    The image paths in the ground_truth dict must be the same as the ones in the
    predictions dict.

    Args:
        ground_truth: The ground truth pose for which to compute metrics in the dataset.
            This should be a dictionary mapping strings (image UIDs, such as image
            paths) to ground truth pose for the image. The pose arrays should be
            in the format (num_individuals, num_bodyparts, 3), where the 3 values are
            x, y and visibility. The ``num_individuals`` corresponds to the number of
            individuals labeled in each image.
        predictions: The predicted poses for which to compute metrics in the dataset.
            This should be a dictionary mapping strings (image UIDs, such as image
            paths) to pose predictions for the image. The pose arrays should be
            in the format (num_predictions, num_bodyparts, 3), where the 3 values are
            x, y and score. The number of predictions can be different to the number of
            ground truth individuals labeled for an image.
        unique_bodypart_gt:
        unique_bodypart_poses: If unique bodyparts are defined for the dataset,
        pcutoff: The threshold to compute the "rmse_cutoff" score (RMSE of all
            predictions with score above the cutoff).
        oks_bbox_margin: The margin to add around keypoints to compute the area for OKS
            computation.
        oks_sigma: The OKS sigma to use to compute pose.

    Returns:
        A dictionary containing keys "rmse", "rmse_cutoff", "mAP" and "mAR" mapping
        to those metrics on the given dataset.

        If unique bodyparts are given, two extra keys "rmse_unique_bodyparts" and
        "rmse_pcutoff_unique_bodyparts" are also returned, containing the metrics for
        the unique bodyparts head.
    """
    data = _prepare_data(ground_truth, predictions)
    rmse, rmse_pcutoff = distance_metrics.compute_rmse(data, pcutoff=pcutoff)
    oks_scores = distance_metrics.compute_oks(
        data=data,
        oks_sigma=oks_sigma,
        oks_bbox_margin=oks_bbox_margin,
    )
    results = dict(rmse=rmse, rmse_pcutoff=rmse_pcutoff, **oks_scores)

    if unique_bodypart_gt is not None:
        # TODO: Should we integrate unique bodyparts to main RMSE?
        assert unique_bodypart_poses is not None
        unique_bpt = _prepare_data(unique_bodypart_gt, unique_bodypart_poses)
        unique_bpt_metrics = distance_metrics.compute_rmse(unique_bpt, pcutoff=pcutoff)
        results["rmse_unique_bodyparts"] = unique_bpt_metrics[0]
        results["rmse_pcutoff_unique_bodyparts"] = unique_bpt_metrics[1]

    return results


def _prepare_data(
    ground_truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Prepares predictions and ground truth pose to compute metrics.

    Only keeps ground truth and predicted assemblies with at least 2 valid keypoints.
    Sets the coordinates for all keypoints that aren't visible (for ground truth,
    visibility <= 0 and for predictions score <= 0) to ``np.nan``.

    Sorts valid predictions by score.

    Args:
        ground_truth: For each image, the GT of shape (n_idv, n_bpt, 3).
        predictions: For each image, the pose predictions of shape (n_pred, n_bpt, 3).

    Returns:
        A list containing (ground truth pose, predicted pose) for each image in the
        dataset, where the predicted pose is sorted from highest to lowest score.
    """
    pose_data = []
    for image, gt in ground_truth.items():
        gt = gt.copy()
        gt[gt[..., 2] <= 0] = np.nan

        # only keep ground truth pose with at least two keypoints
        gt_mask = np.sum(np.all(~np.isnan(gt), axis=-1), axis=-1) > 1
        gt = gt[gt_mask]

        pred = predictions[image][..., :3].copy()  # PAF have 5 values; keep xy + score
        pred[pred[..., 2] <= 0] = np.nan

        # only keep predicted pose with at least two keypoints
        pred_mask = np.sum(np.all(~np.isnan(pred), axis=-1), axis=-1) > 1
        pred = pred[pred_mask]

        scores = np.nanmean(pred[:, :, 2], axis=-1)
        pred_order = np.argsort(-scores, kind="mergesort")
        pose_data.append((gt, pred[pred_order]))

    return pose_data
