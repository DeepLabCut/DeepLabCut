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
    """Summary:
    Hungarian algorithm predicted individuals to ground truth ones, using root mean squared error (rmse). The function provides a way to
    match predicted individuals to ground truth individuals based on the rmse distance between their corresponding
    keypoints. This algorithm is used to find the optimal matching, taking into account the potential missing animal.

    Raises:
        ValueError: if `gt_kpts.shape != pred_kpts.shape`

    Args:
        pred_kpts: predicted keypoints for each animal. The shape of the array is (num_animals, num_keypoints, 3):
            num_animals: number of animals
            num_keypoints: number of keypoints
            3: (x,y,score) coordinates of each keypoint
        gt_kpts: ground truth keypoints for each animal. The shape of the array is (num_animals, num_keypoints, 2):
            num_animals: number of animals
            num_keypoints: number of keypoints
            2: (x,y) coordinates of each keypoint

    Returns:
        col_ind (np.array): array of the individuals indices for prediction
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

    distance_matrix = np.zeros((num_animals_gt, num_animals))
    for g in range(num_animals_gt):
        for p in range(num_animals):
            distance_matrix[g, p] = np.nansum(
                (gt_kpts_without_ctr[g] - pred_kpts[p, :, :2]) ** 2
            )

    _, col_ind = linear_sum_assignment(distance_matrix)
    # if animals are missing in the frame, the predictions corresponding to nothing are not shuffled
    col_ind = extend_col_ind(col_ind, num_animals)

    return col_ind


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
