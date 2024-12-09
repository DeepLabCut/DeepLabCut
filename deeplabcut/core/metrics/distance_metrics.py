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
"""Implementations of methods to compute distance metrics such as RMSE or OKS"""
from __future__ import annotations

import numpy as np

import deeplabcut.core.metrics.matching as matching
from deeplabcut.core.crossvalutils import find_closest_neighbors
from deeplabcut.core.inferenceutils import calc_object_keypoint_similarity


def compute_oks_matrix(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    oks_sigma: float,
    oks_bbox_margin: float = 0.0,
) -> np.ndarray:
    """Computes the OKS score for each (prediction, gt) pair in an image

    Args:
        ground_truth: The GT poses for an image, shape (n_individuals, n_kpts, 2)
        predictions: The predicted poses in the image, shape (n_pred, n_kpts, 2)
        oks_sigma: The sigma value to use to compute OKS
        oks_bbox_margin: The margin to add around keypoints when computing the area.
            FIXME(niels) We should allow the use of ground truth bboxes to get area

    Returns:
        A matrix of shape (n_pred, n_kpts) where entry (i, j) is the OKS between
        prediction i and ground truth j.
    """
    oks_matrix = np.zeros((len(predictions), len(ground_truth)))
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            oks_matrix[pred_idx, gt_idx] = calc_object_keypoint_similarity(
                pred[:, :2],
                gt[:, :2],
                sigma=oks_sigma,
                margin=oks_bbox_margin,
            )

    return oks_matrix


def compute_oks(
    data: list[tuple[np.ndarray, np.ndarray]],
    oks_bbox_margin: float = 0.0,
    oks_sigma: float = 0.1,
    oks_thresholds: np.ndarray | None = None,
    oks_recall_thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes the OKS for pose at different thresholds.

    Args:
        data: The data for which to compute OKS mAP: a list containing (gt_poses,
            predicted_poses) tuples, where gt_pose is an array of shape
            (num_gt_individuals, num_bpts, 3) and predicted_poses is an array of shape
            (num_predictions, num_bpts, 3). For the GT, the 3 coordinates are (x, y,
            visibility) while for the pose they are (x, y, confidence score).
        oks_sigma: The OKS sigma to use to compute pose.
        oks_bbox_margin: The margin to add around keypoints to compute the area for OKS
            computation.
        oks_thresholds: The OKS thresholds at which to compute AP. If None, defaults to
            (0.5, 0.55, 0.6, ..., 0.9, 0.95).
        oks_recall_thresholds: The recall thresholds to use to compute mAP. If None,
            defaults to the same default values used in pycocotools.

    Returns:
        A dictionary containing mAP and mAR scores.
    """
    if oks_thresholds is None:
        oks_thresholds = np.linspace(0.5, 0.95, 10)

    if oks_recall_thresholds is None:
        oks_recall_thresholds = np.linspace(
            start=0.0,
            stop=1.00,
            num=int(np.round((1.00 - 0.0) / 0.01)) + 1,
            endpoint=True,
        )

    total_gt = 0
    pose_data = []
    for gt, pred in data:
        # filter data to only keep individuals with at least 2 valid keypoints
        gt = gt[np.sum(np.all(~np.isnan(gt), axis=-1), axis=-1) > 1]
        pred = pred[np.sum(np.all(~np.isnan(pred), axis=-1), axis=-1) > 1]

        oks_matrix = compute_oks_matrix(
            gt[:, :, :2],
            pred[:, :, :2],
            oks_sigma=oks_sigma,
            oks_bbox_margin=oks_bbox_margin,
        )

        total_gt += len(gt)
        pose_data.append((gt, pred, oks_matrix))

    precisions, recalls = [], []
    for oks_threshold in oks_thresholds:
        matches = []
        for gt, pred, oks_matrix in pose_data:
            image_matches = matching.match_greedy_oks(
                gt,
                pred,
                oks_matrix=oks_matrix,
                oks_threshold=oks_threshold,
            )
            matches.extend(image_matches)

        if len(matches) == 0:  # no predictions -> precision 0, recall 0
            return {"mAP": 0, "mAR": 0}

        scores = np.asarray([m.score for m in matches])
        match_order = np.argsort(-scores, kind="mergesort")
        oks_values = np.asarray([m.oks for m in matches])
        oks_values = oks_values[match_order]

        tp = np.cumsum(oks_values >= oks_threshold)
        fp = np.cumsum(oks_values < oks_threshold)
        rc = tp / total_gt
        pr = tp / (fp + tp + np.spacing(1))
        recall = rc[-1]

        # Guarantee precision decreases monotonically, see
        # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        inds_rc = np.searchsorted(rc, oks_recall_thresholds, side="left")
        precision = np.zeros(inds_rc.shape)
        valid = inds_rc < len(pr)
        precision[valid] = pr[inds_rc[valid]]

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    return {
        "mAP": 100 * precisions.mean().item(),
        "mAR": 100 * recalls.mean().item(),
    }


def compute_rmse(
    data: list[tuple[np.ndarray, np.ndarray]],
    single_animal: bool,
    pcutoff: float,
    oks_bbox_margin: float = 0.0,
) -> tuple[float, float]:
    """Computes the RMSE for pose predictions.

    Single animal RMSE is computed by simply calculating the distance between each
    ground truth keypoint and the corresponding prediction.

    Multi-animal RMSE is computed differently: predictions are first matched to ground
    truth individuals using greedy OKS matching. RMSE is then computed only between
    predictions and the ground truth pose they are matched to, only when the OKS is
    non-zero (greater than a small threshold). Predictions that cannot be matched to
    any ground truth with non-zero OKS are not used to compute RMSE.

    Args:
        data: The data for which to compute RMSE. This is a list containing (gt_poses,
            predicted_poses), where gt_pose is an array of shape (num_gt_individuals,
            num_bpts, 3) and predicted_poses is an array of shape (num_predictions,
            num_bpts, 3). For the GT, the 3 coordinates are (x, y, visibility) while for
            the pose they are (x, y, confidence score).
        single_animal: Whether this is a single animal dataset.
        pcutoff: The p-cutoff to use to compute RMSE.
        oks_bbox_margin: When single_animal is False, predictions are matched to GT
            using OKS. This is the margin used to apply when computing the bbox from
            the pose to compute OKS.

    Returns:
        The RMSE and RMSE after removing all detections with a score below the pcutoff.

    Raises:
        AssertionError
    """
    matches = []
    for gt, pred in data:
        if single_animal:
            if gt.shape[0] > 1 or pred.shape[0] > 1:
                raise ValueError(
                    "At most 1 individual and 1 prediction can be given when computing "
                    f"single animal RMSE. Found gt={gt.shape}, pred={pred.shape}"
                )

            image_matches = []
            if gt.shape[0] == 1 and pred.shape[0] == 1:
                match = matching.PotentialMatch.from_pose(pred[0])
                match.match(gt[0], oks=float("nan"))  # OKS not needed for RMSE
                image_matches.append(match)
        else:
            oks_matrix = compute_oks_matrix(
                gt[:, :, :2],
                pred[:, :, :2],
                oks_sigma=0.1,
                oks_bbox_margin=oks_bbox_margin,
            )
            image_matches = matching.match_greedy_oks(
                gt,
                pred,
                oks_matrix=oks_matrix,
                oks_threshold=1e-6,
            )

        matches.extend(image_matches)

    rmse, rmse_cutoff = float("nan"), float("nan")
    if len(matches) == 0:
        return rmse, rmse_cutoff

    pixel_errors = np.stack([m.pixel_errors() for m in matches])
    if np.any(~np.isnan(pixel_errors)):
        rmse = np.nanmean(pixel_errors).item()

        keypoint_scores = np.stack([m.keypoint_scores() for m in matches])
        pixel_errors_cutoff = pixel_errors[keypoint_scores >= pcutoff]
        if np.any(~np.isnan(pixel_errors_cutoff)):
            rmse_cutoff = np.nanmean(pixel_errors_cutoff).item()

    return rmse, rmse_cutoff


def compute_detection_rmse(
    data: list[tuple[np.ndarray, np.ndarray]],
    pcutoff: float,
) -> tuple[float, float]:
    """Computes the detection RMSE for pose predictions.

    The detection RMSE score does not take individual assemblies into account. It only
    judges the performance of the detections, matching each predicted keypoint to the
    closest ground truth for each bodypart.

    This is the same way multi-animal RMSE was computed in DeepLabCut 2.X.

    Args:
        data: The data for which to compute RMSE. This is a list containing (gt_poses,
            predicted_poses), where gt_pose is an array of shape (num_gt_individuals,
            num_bpts, 3) and predicted_poses is an array of shape (num_predictions,
            num_bpts, 3). For the GT, the 3 coordinates are (x, y, visibility) while for
            the pose they are (x, y, confidence score).
        pcutoff: The p-cutoff to use to compute RMSE.

    Returns:
        The detection RMSE and detection RMSE after removing all detections with a
        score below the pcutoff.
    """
    distances = []
    scores = []
    for image_gt, image_pred in data:
        image_gt = image_gt.transpose((1, 0, 2))  # to (num_bpts, num_gt_individuals, 3)
        image_pred = image_pred.transpose((1, 0, 2))  # to (num_bpts, num_pred, 3)

        for bpt_index, (bpt_gt, bpt_pred) in enumerate(zip(image_gt, image_pred)):
            # filter NaNs and invalid values
            bpt_gt = bpt_gt[~np.any(np.isnan(bpt_gt), axis=1)]
            bpt_pred = bpt_pred[~np.any(np.isnan(bpt_pred), axis=1)]
            if len(bpt_gt) == 0 or len(bpt_pred) == 0:
                continue

            # assignment of predicted bodyparts to ground truth
            neighbors = find_closest_neighbors(bpt_gt, bpt_pred, k=3)
            for gt_index, pred_index in enumerate(neighbors):
                if pred_index != -1:
                    gt = bpt_gt[gt_index]
                    pred = bpt_pred[pred_index]
                    distances.append(np.linalg.norm(gt[:2] - pred[:2]))
                    scores.append(bpt_pred[pred_index, 2])

    rmse, rmse_cutoff = float("nan"), float("nan")
    if len(distances) == 0:
        return rmse, rmse_cutoff

    distances = np.stack(distances)
    if np.any(~np.isnan(distances)):
        rmse = np.nanmean(distances).item()

        keypoint_scores = np.stack(scores)
        pixel_errors_cutoff = distances[keypoint_scores >= pcutoff]
        if np.any(~np.isnan(pixel_errors_cutoff)):
            rmse_cutoff = np.nanmean(pixel_errors_cutoff).item()

    return rmse, rmse_cutoff
