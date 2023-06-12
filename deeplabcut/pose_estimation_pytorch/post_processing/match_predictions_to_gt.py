import numpy as np
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import calc_object_keypoint_similarity
from scipy.optimize import linear_sum_assignment

def rmse_match_prediction_to_gt(pred_kpts: np.array, gt_kpts: np.array, individual_names: list):
    '''
        Hungarian algorithm predicted individuals to ground truth ones, using rmse
        
    Arguments
    ---------
        pred_kpts: (num_animals, num_keypoints, 3)
        gt_kpts: (num_animals, num_keypoints(+1 if with center), 2)
        individual_names: names of individuals

    Output
    ------
        row_ind: array of the individuals indexes for prediction
    '''

    num_animals, num_keypoints, _ = pred_kpts.shape
    if num_keypoints + 1 == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts[:, :-1, :].copy()
    elif num_keypoints == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts.copy()
    else:
        raise ValueError('Shape mismatch between ground truth and predictions')

    # Computation of the number of annotated animals in the ground truth
    num_animals_gt = num_animals
    for animal_index in range(num_animals):
        if (gt_kpts_without_ctr[animal_index] < 0).all():
            num_animals_gt -= 1

    distance_matrix = np.zeros((num_animals_gt, num_animals))
    for g in range(num_animals_gt):
        for p in range(num_animals):
            distance_matrix[g, p] = np.linalg.norm(gt_kpts_without_ctr[g] - pred_kpts[p, :, :2])

    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # if animals are missing in the frame, the predictions corresponding to nothing are not shuffled
    col_ind = extend_col_ind(col_ind, num_animals)
    
    return col_ind


def oks_match_prediction_to_gt(pred_kpts: np.array, gt_kpts: np.array, individual_names: list):
    '''
        Hungarian algorithm predicted individuals to ground truth ones, using oks
        
    Arguments
    ---------
        pred_kpts: (num_animals, num_keypoints, 3)
        gt_kpts: (num_animals, num_keypoints(+1 if with center), 2)
        individual_names: names of individuals

    Output
    ------
        row_ind: array of the individuals indexes for prediction
    '''

    num_animals, num_keypoints, _ = pred_kpts.shape
    if num_keypoints + 1 == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts[:, :-1, :].copy()
    elif num_keypoints == gt_kpts.shape[1]:
        gt_kpts_without_ctr = gt_kpts.copy()
    else:
        raise ValueError('Shape mismatch between ground truth and predictions')

    # Computation of the number of annotated animals in the ground truth
    num_animals_gt = num_animals
    for animal_index in range(num_animals):
        if (gt_kpts_without_ctr[animal_index] < 0).all():
            num_animals_gt -= 1

    oks_matrix = np.zeros((num_animals_gt, num_animals))
    gt_kpts_without_ctr[gt_kpts_without_ctr < 0]  = np.nan # non visible keypoints should be nan to use calc_oks
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
                symmetric_kpts=None #TODO take into account symmetric keypoints
            )

    row_ind, col_ind = linear_sum_assignment(oks_matrix, maximize=True)
    # if animals are missing in the frame, the predictions corresponding to nothing are not shuffled
    col_ind = extend_col_ind(col_ind, num_animals)
    
    return col_ind

def extend_col_ind(col_ind, num_animals):
    existing_cols = set(col_ind)  # Convert the array to a set for faster lookup
    missing_cols = [num for num in range(num_animals) if num not in existing_cols]
    extended_array = np.concatenate((col_ind, missing_cols)).astype(int)
    return extended_array