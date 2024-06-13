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

import numpy as np
from scipy.optimize import linear_sum_assignment

from deeplabcut.core.inferenceutils import (
    calc_object_keypoint_similarity,
)


def rmse_match_prediction_to_gt(
    pred_kpts: np.ndarray, gt_kpts: np.ndarray
) -> np.ndarray:
    """
    Hungarian algorithm predicted individuals to ground truth ones, using root mean
    squared error (rmse). The function provides a way to match predicted individuals to
    ground truth individuals based on the rmse distance between their corresponding
    keypoints. This algorithm is used to find the optimal matching, taking into account
    the potential missing animal.

    Raises:
        ValueError: if `gt_kpts.shape != pred_kpts.shape`

    Args:
        pred_kpts: shape (num_individuals, num_keypoints, 3), ground truth keypoints for
            an image, where the 3 values are (x,y,score) for each keypoint
        gt_kpts: shape (num_individuals, num_keypoints, 3), ground truth keypoints for
            an image, where the 3 values are (x,y,visibility) for each keypoint

    Returns:
        col_ind: array of the individuals indices for prediction
    """
    num_pred, num_keypoints, _ = pred_kpts.shape
    num_idv, num_keypoints_gt, _ = gt_kpts.shape
    if num_keypoints + 1 == num_keypoints_gt:
        gt_kpts = gt_kpts[:, :-1, :].copy()
    elif num_keypoints == num_keypoints_gt:
        gt_kpts = gt_kpts.copy()
    else:
        raise ValueError("Shape mismatch between ground truth and predictions")

    if num_pred != num_idv:
        raise ValueError(
            "Must have the same number of GT and predicted individuals, found "
            f"pred_kpts={pred_kpts.shape} and gt_kpts={gt_kpts.shape}"
        )

    valid_gt = np.any(gt_kpts[..., 2] > 0, axis=1)
    valid_gt_indices = np.nonzero(valid_gt)[0]
    if len(valid_gt_indices) == 0:
        return np.arange(num_idv)

    valid_pred = np.any(pred_kpts[..., 2] > 0, axis=1)
    valid_pred_indices = np.nonzero(valid_pred)[0]
    if len(valid_pred_indices) == 0:
        return np.arange(num_idv)

    distance_matrix = np.full((len(valid_gt_indices), len(valid_pred_indices)), np.inf)
    for i, gt_idx in enumerate(valid_gt_indices):
        gt_idv = gt_kpts[gt_idx]
        mask = gt_idv[:, 2] > 0
        for j, pred_idx in enumerate(valid_pred_indices):
            pred_idv = pred_kpts[pred_idx]
            d = (gt_idv[mask, :2] - pred_idv[mask, :2]) ** 2
            distance_matrix[i, j] = np.nanmean(d)

    _, col_ind = linear_sum_assignment(distance_matrix)  # len == len(valid_gt_indices)

    gt_idx_to_pred_idx = {
        valid_gt_indices[valid_gt_index]: valid_pred_indices[valid_pred_index]
        for valid_gt_index, valid_pred_index in enumerate(col_ind)
    }
    matched_pred = {valid_pred_indices[i] for i in col_ind}
    unmatched_pred = [i for i in range(num_idv) if i not in matched_pred]
    next_unmatched = 0
    col_ind = []
    for gt_index in range(num_idv):
        if gt_index in gt_idx_to_pred_idx:
            col_ind.append(gt_idx_to_pred_idx[gt_index])
        else:
            col_ind.append(unmatched_pred[next_unmatched])
            next_unmatched += 1

    return np.array(col_ind)


def oks_match_prediction_to_gt(
    pred_kpts: np.array, gt_kpts: np.array, individual_names: list
) -> np.array:
    """Summary:
    Hungarian algorithm predicted individuals to ground truth ones, using object keypoint similarity (oks).
    Oks measures the accuracy of predicted keypoints compared to ground truth keypoints.
    More information about oks can be found in cocodataset (https://cocodataset.org/#keypoints-eval).

    Args:
        pred_kpts: Predicted keypoints for each animal. The shape of the array is (num_animals, num_keypoints, 3):
            num_animals: Number of animals.
            num_keypoints: Number of keypoints.
            3: (x, y, score) coordinates of each keypoint.
        gt_kpts: Ground truth keypoints for each animal. The shape of the array is (num_animals, num_keypoints(+1 if with center), 2):
            num_animals: Number of animals.
            num_keypoints: Number of keypoints.
        individual_names: names of individuals

    Returns:
        col_ind: Array of the individual indexes for prediction.

    Examples:
        input:
            pred_kpts = np.array(...)
            gt_kpts = np.array(...)
            individual_names = [...]
        output:
            col_ind = np.array([...])
    """

    num_animals, num_keypoints, _ = pred_kpts.shape
    if num_keypoints + 1 == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts[:, :-1, :].copy()
    elif num_keypoints == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts.copy()
    else:
        raise ValueError("Shape mismatch between ground truth and predictions")

    # Computation of the number of annotated animals in the ground truth
    num_animals_gt = num_animals
    for animal_index in range(num_animals):
        if (gt_kpts_without_ctr[animal_index] < 0).all():
            num_animals_gt -= 1

    oks_matrix = np.zeros((num_animals_gt, num_animals))
    gt_kpts_without_ctr[
        gt_kpts_without_ctr < 0
    ] = np.nan  # non visible keypoints should be nan to use calc_oks
    idx_gt = -1
    for g in range(num_animals):
        if np.isnan(gt_kpts_without_ctr[g]).all():
            continue
        else:
            idx_gt += 1
        for p in range(num_animals):
            oks_matrix[idx_gt, p] = calc_object_keypoint_similarity(
                pred_kpts[p, :, :2],
                gt_kpts_without_ctr[g],
                0.1,
                margin=0,
                symmetric_kpts=None,  # TODO take into account symmetric keypoints
            )

    row_ind, col_ind = linear_sum_assignment(oks_matrix, maximize=True)
    # if animals are missing in the frame, the predictions corresponding to nothing are not shuffled
    col_ind = extend_col_ind(col_ind, num_animals)

    return col_ind


def extend_col_ind(col_ind: np.array, num_animals: int) -> np.array:
    """Summary:
    Extends the column indices of a 1D array, col_ind, by adding any missing column indices from 0 to num_animals-1.

    Args:
        col_ind: 1D array of column indices
        num_animals: total number of animals

    Returns:
        extended_array: extended 1D array of column indices

    Examples:
        input:
            col_ind =
            num_animals = 5
        output:
            extended_array =
    """
    existing_cols = set(col_ind)  # Convert the array to a set for faster lookup
    missing_cols = [num for num in range(num_animals) if num not in existing_cols]
    extended_array = np.concatenate((col_ind, missing_cols)).astype(int)
    return extended_array
