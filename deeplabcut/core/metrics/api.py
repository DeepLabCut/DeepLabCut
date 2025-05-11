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
    single_animal: bool = False,
    unique_bodypart_gt: dict[str, np.ndarray] | None = None,
    unique_bodypart_poses: dict[str, np.ndarray] | None = None,
    pcutoff: float = -1,
    oks_bbox_margin: int = 0,
    oks_sigma: float | np.ndarray = 0.1,
    per_keypoint_rmse: bool = False,
    compute_detection_rmse: bool = True,
) -> dict:
    """Computes pose estimation performance metrics

    Given ground truth pose labels and predictions on a dataset, computes RMSE and pose
    mAP/mAR using OKS.

    The image paths in the ground_truth dict must be the same as the ones in the
    predictions dict.

    Single animal RMSE is computed by simply calculating the Euclidean distance between
    each ground truth keypoint and the corresponding prediction.

    Multi-animal RMSE is computed differently: predictions are first matched to ground
    truth individuals using greedy OKS matching. OKS (or object keypoint similarity) is
    a similarity metric for keypoints (you can read more about it and its definition
    here: https://cocodataset.org/#keypoints-eval). RMSE is then computed only between
    predictions and the ground truth pose they are matched to, only when the OKS is
    greater than a small threshold. Predictions that cannot be matched to any ground
    truth with non-zero OKS are not used to compute RMSE.

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
        single_animal: Whether the metrics are being computed on a single-animal or
            multi-animal dataset. This has an impact on RMSE computation.
        unique_bodypart_gt: If unique bodyparts are defined for the dataset, they should
            be contained in this dict in the same format as the ``ground_truth`` dict.
        unique_bodypart_poses: If unique bodyparts are defined for the dataset, the
            predictions should be contained in this dict in the same format as the
            ``predictions`` dict.
        pcutoff: The threshold to compute the "rmse_cutoff" score (RMSE of all
            predictions with score above the cutoff).
        oks_bbox_margin: The margin to add around keypoints to compute the area for OKS
            computation.
        oks_sigma: The OKS sigma to use to compute pose.
        per_keypoint_rmse: Compute per-keypoint RMSE values.
        compute_detection_rmse: Computes detection RMSE (without animal assembly) if the
            predictions are from a multi-animal model.

    Returns:
        A dictionary containing keys "rmse", "rmse_cutoff", "mAP" and "mAR" mapping
        to those metrics on the given dataset.

        If unique bodyparts are given, two extra keys "rmse_unique_bodyparts" and
        "rmse_pcutoff_unique_bodyparts" are also returned, containing the metrics for
        the unique bodyparts head.

        If `per_keypoint_evaluation=True`, "keypoint_rmse", "keypoint_rmse_cutoff" (and
        optionally "unique_keypoint_rmse" and "unique_keypoint_rmse_cutoff") keys are
        added, containing a list of floats representing the RMSE for each keypoint.

    Examples:
        >>> # Define the p-cutoff, prediction, and target DataFrames
        >>> pcutoff = 0.5
        >>> ground_truth = {"img0": np.array([[[1.0, 1.0, 2.0], ...], ...]), ...}
        >>> predictions = {"img0": np.array([[[2.0, 1.0, 0.4], ...], ...]), ...}
        >>> scores = compute_metrics(ground_truth, predictions, pcutoff=pcutoff)
        >>> print(scores)
        {
            "rmse": 1.0,
            "rmse_pcutoff": 0.0,
            'mAP': 84.2,
            'mAR': 74.5
        }  # Sample output scores
    """
    data = prepare_evaluation_data(ground_truth, predictions)
    oks_scores = distance_metrics.compute_oks(
        data=data,
        oks_sigma=oks_sigma,
        oks_bbox_margin=oks_bbox_margin,
    )

    data_unique = None
    if unique_bodypart_gt is not None:
        assert unique_bodypart_poses is not None
        data_unique = prepare_evaluation_data(unique_bodypart_gt, unique_bodypart_poses)

    rmse_scores = distance_metrics.compute_rmse(
        data,
        single_animal,
        pcutoff,
        data_unique=data_unique,
        per_keypoint_results=per_keypoint_rmse,
    )
    results = dict(**rmse_scores, **oks_scores)

    if compute_detection_rmse and not single_animal:
        det_rmse, det_rmse_p = distance_metrics.compute_detection_rmse(
            data, pcutoff, data_unique=data_unique,
        )
        results["rmse_detections"] = det_rmse
        results["rmse_detections_pcutoff"] = det_rmse_p

    return results


def prepare_evaluation_data(
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

        # only keep ground truth pose with at least one keypoint
        gt_mask = np.any(np.all(~np.isnan(gt), axis=-1), axis=-1)
        gt = gt[gt_mask]

        pred = predictions[image][..., :3].copy()  # PAF have 5 values; keep xy + score
        pred[pred[..., 2] < 0] = np.nan

        # only keep predicted pose with at least two keypoints
        pred_mask = np.any(np.all(~np.isnan(pred), axis=-1), axis=-1)
        pred = pred[pred_mask]

        scores = np.nanmean(pred[:, :, 2], axis=-1)
        pred_order = np.argsort(-scores, kind="mergesort")
        pose_data.append((gt, pred[pred_order]))

    return pose_data
