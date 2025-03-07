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
"""Implementations of methods to compute identity prediction accuracy"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from deeplabcut.core.crossvalutils import find_closest_neighbors


def compute_identity_scores(
    individuals: list[str],
    bodyparts: list[str],
    predictions: dict[str, np.ndarray],
    identity_scores: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    FIXME: With DLCRNet all heatmap "peaks" above 0.01 were kept, with 1 keypoint and
     1 identity score map per peak. Then, for each ground truth keypoint, we selected
     the prediction closest to it, and evaluated the identity score in that position.
     This is no longer the case, as we're now evaluating after assembly. So we only
     have num_individuals assemblies.

    Args:
        individuals:
        bodyparts:
        predictions: (num_assemblies, num_bodyparts, 3)
        identity_scores: (num_assemblies, num_bodyparts, num_individuals)
        ground_truth: (num_individuals, num_bodyparts, 3)

    Returns:

    """
    if not len(predictions) == len(ground_truth):
        raise ValueError("Mismatch between number of predictions and ground truth")

    all_bpts = np.asarray(len(individuals) * bodyparts)
    ids = np.full((len(predictions), len(all_bpts), 2), np.nan)
    for i, (image, pred) in enumerate(predictions.items()):
        for j in range(len(individuals)):
            for k in range(len(bodyparts)):
                bpt_idx = len(bodyparts) * j + k
                ids[i, bpt_idx, 0] = j

        # set keypoints that aren't visible to NaN
        gt = ground_truth[image].copy()
        gt[gt[..., 2] <= 0, :2] = np.nan
        gt = gt[..., :2]

        id_scores = identity_scores[image]

        # reorder to (bodypart, individual, ...)
        gt = gt.transpose((1, 0, 2))
        pred = pred.transpose((1, 0, 2))[..., :2]
        id_scores = id_scores.transpose((1, 0, 2))
        for bpt, bpt_gt, bpt_pred, bpt_id_scores in zip(bodyparts, gt, pred, id_scores):
            # assign ground truth keypoints to the closest prediction, so the ID score
            # is the closest possible to the ID score computed with "ground truth"
            indices_gt = np.flatnonzero(np.all(~np.isnan(bpt_gt), axis=1))

            # Remove NaN predictions from the bodypart predictions
            indices_pred = np.all(np.isfinite(bpt_pred), axis=1)
            bpt_pred = bpt_pred[indices_pred]
            bpt_id_scores = bpt_id_scores[indices_pred]

            neighbors = find_closest_neighbors(bpt_gt[indices_gt], bpt_pred, k=3)
            found = neighbors != -1
            indices = np.flatnonzero(all_bpts == bpt)
            # Get the predicted identity of each bodypart by taking the argmax
            ids[i, indices[indices_gt[found]], 1] = np.argmax(
                bpt_id_scores[neighbors[found]], axis=1
            )

    ids = ids.reshape((len(predictions), len(individuals), len(bodyparts), 2))
    results = {}
    for i, bpt in enumerate(bodyparts):
        temp = ids[:, :, i].reshape((-1, 2))
        valid = np.isfinite(temp).all(axis=1)
        y_true, y_pred = temp[valid].T
        results[f"{bpt}_accuracy"] = accuracy_score(y_true, y_pred)

    return results
