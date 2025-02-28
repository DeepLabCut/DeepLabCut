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
"""Functions to assign identity to predictions from an identity head"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_identity(
    predictions: np.ndarray, identity_scores: np.ndarray
) -> np.ndarray:
    """
    Args:
        predictions: Pose predictions for an image, with shape (num_individuals,
            num_bodyparts, 3)
        identity_scores: Identity predictions for keypoints in an image, of shape
            (num_individuals, num_bodyparts, num_individuals).

    Returns:
        The ordering to use to match predictions to identities.
    """
    if not len(predictions) == len(identity_scores):
        raise ValueError(
            "There are not the same number of predictions as identity scores"
            f" ({len(predictions)} != {len(identity_scores)}"
        )

    # average of ID scores, weighted by keypoint confidence
    pose_conf = predictions[:, :, 2:3]
    cost_matrix = np.mean(pose_conf * identity_scores, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    new_order = np.zeros_like(row_ind)
    for old_pos, new_pos in zip(row_ind, col_ind):
        new_order[new_pos] = old_pos

    return new_order
