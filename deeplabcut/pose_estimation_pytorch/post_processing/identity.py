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
    predictions: list[np.ndarray],
    identity_scores: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Args:
        predictions: shape (num_individuals, num_bodyparts, 3)
        identity_scores: shape (num_individuals, num_bodyparts, num_individuals)

    Returns:
        predictions with assigned identity, of shape (num_individuals, num_bodyparts, 3)
    """
    if not len(predictions) == len(identity_scores):
        raise ValueError(
            "There are not the same number of predictions as identity scores"
            f" ({len(predictions)} != {len(identity_scores)}"
        )

    predictions_with_identity = []
    for pred, scores in zip(predictions, identity_scores):
        cost_matrix = np.product(scores, axis=1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        new_order = np.zeros_like(row_ind)
        for old_pos, new_pos in zip(row_ind, col_ind):
            new_order[new_pos] = old_pos

        predictions_with_identity.append(pred[new_order])

    return predictions_with_identity
