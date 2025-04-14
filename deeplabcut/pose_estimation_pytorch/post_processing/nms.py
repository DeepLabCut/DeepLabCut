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
"""Methods for non-maximum suppression of detected poses."""
import numpy as np

from deeplabcut.core.inferenceutils import calc_object_keypoint_similarity


def nms_oks(
    predictions: np.ndarray,
    oks_threshold: float,
    oks_sigmas: float | np.ndarray = 0.1,
    oks_margin: float = 1.0,
    score_threshold: float | None = None,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Implementation of NMS using OKS.

    Args:
        predictions: The predicted poses, of shape (num_predictions, num_keypoints, 3).
        oks_threshold: The threshold for NMS. Keeps predictions for which the OKS score
            is below this threshold.
        oks_sigmas: The sigmas to use to compute OKS scores.
        oks_margin: The margin to add around keypoints when computing area.
        score_threshold: If not None, computes NMS using only keypoints for which the
            score is above this threshold.
        order: If predictions should be sorted by another means than score, the order
            to use in NMS.

    Returns:
        An array of length num_predictions indicating which keypoints should be kept.
    """
    if len(predictions) == 0:
        return np.zeros(0, dtype=bool)
    elif len(predictions) == 1:
        return np.ones(1, dtype=bool)

    predictions = predictions.copy()

    # mask keypoints with score below the threshold
    if score_threshold is None:
        score_threshold = 0.0
    predictions[predictions[:, :, 2] < score_threshold] = np.nan

    # get visibility masks for the keypoints and individuals
    kpt_vis = np.all(~np.isnan(predictions), axis=-1)
    idv_vis = np.sum(kpt_vis, axis=-1) > 1  # need at least 2 keypoints to compute OKS

    # if no keypoints match the visibility criteria, mask all
    if np.sum(idv_vis) == 0:
        return np.zeros(len(predictions), dtype=bool)

    # mask keypoints that aren't visible
    predictions[~kpt_vis] = np.nan

    if order is None:
        # compute scores for each individual
        scores = np.zeros(len(predictions))
        scores[idv_vis] = np.nanmean(predictions[idv_vis, :, 2], axis=-1)

        # only compute OKS for non-zero score poses
        order = scores.argsort()[::-1]
        order = order[scores[order] > 0]

    # NMS suppression
    keep = np.zeros(len(predictions), dtype=bool)
    while len(order) > 0:
        i = order[0]
        order = order[1:]
        keep[i] = True

        oks_scores = [
            calc_object_keypoint_similarity(
                predictions[i],
                predictions[j],
                sigma=oks_sigmas,
                margin=oks_margin,
            )
            for j in order
        ]
        to_keep = [s < oks_threshold and not np.isnan(s) for s in oks_scores]
        order = [idx for idx, kept in zip(order, to_keep) if kept]

    return keep
