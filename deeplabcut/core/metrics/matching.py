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
"""Algorithms to match predictions to ground truth labels"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PotentialMatch:
    """A potential match between predicted pose and ground truth pose.

    Args:
        pose: An array of shape (num_bodyparts, 3)
        score: The score for the prediction. This could be the mean of the confidence
            score for each bodypart, or another value representing how confident the
            model is that this assembly is correct.
        gt: None if no ground truth pose was matched to the prediction. If defined, the
            ground truth to which the prediction is matched. It should be of shape
            (num_bodyparts, 3), where the 3 values are x, y and visibility.
        oks: The OKS score between the pose and the ground truth.
    """

    pose: np.ndarray
    score: float
    gt: np.ndarray | None = None
    oks: float = 0.0

    def keypoint_scores(self) -> np.ndarray:
        """Returns: The confidence score for each bodypart in the predicted pose."""
        return self.pose[:, 2].copy()

    def pixel_errors(self) -> np.ndarray:
        """
        Returns:
            The distance (in pixels) between each predicted and ground truth bodypart.
            If this prediction is unmatched, returns an array of length num_bodyparts
            containing all NaNs.
        """
        if self.gt is None:
            return np.full(len(self.pose), np.nan)

        return np.linalg.norm(self.pose[:, :2] - self.gt[:, :2], axis=1)

    def match(self, gt: np.ndarray, oks: float) -> None:
        """Adds a ground truth match to this PotentialMatch

        Args:
            gt: The ground truth to which the prediction is matched. The ground truth
                pose should be of shape (num_bodyparts, 3), where the 3 values are x, y
                and visibility.
            oks: The OKS similarity between the ground truth and this.
        """
        self.gt = gt
        self.oks = oks

    @classmethod
    def from_pose(cls, pose: np.ndarray) -> "PotentialMatch":
        assert len(pose.shape) == 2  # Must be pose for a single individual
        scores = pose[:, 2]
        if np.all(np.isnan(scores)):
            raise ValueError(
                "Cannot create a Match from a pose prediction where all scores are nan "
                f"(pose={pose})"
            )

        return PotentialMatch(pose=pose, score=np.nanmean(scores).item())


def match_greedy_oks(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    oks_matrix: np.ndarray,
    oks_threshold: float = 0.0,
) -> list[PotentialMatch]:
    """Greedy matching of ground truth individuals to predicted individuals using OKS

    This is done in the same way as done in pycocotools. The predictions must be sorted
    by score before being passed to this function.

    Args:
        ground_truth: The ground truth labels for an image, of shape (n_idv, n_bpt, 2)
        predictions: The predictions for an image, of shape (n_idv, n_bpt, 2)
        oks_matrix: A matrix of shape (n_pred, n_kpts) where entry (i, j) is the OKS
            between prediction i and ground truth j.
        oks_threshold: The min. OKS for a prediction to be matched to a GT pose

    Returns:
        A list containing a PotentialMatch for each predicted pose in the given
        predictions.
    """
    matches = [PotentialMatch.from_pose(pose=pred) for pred in predictions]
    matched_gt_indices = set()
    for idx, pred in enumerate(predictions):
        oks = oks_matrix[idx]
        if np.all(np.isnan(oks)):
            continue

        ind_best = np.nanargmax(oks)

        # if this gt already matched, continue
        if ind_best in matched_gt_indices:
            continue

        # Only match the pred to the GT if the OKS value is above a given threshold
        if oks[ind_best] < oks_threshold:
            continue

        matched_gt_indices.add(ind_best)
        matches[idx].match(gt=ground_truth[ind_best], oks=oks[ind_best])

    return matches


def match_greedy_rmse(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    keep_assemblies: bool = True,
) -> list[PotentialMatch]:
    """Greedy matching of ground truth individuals to predicted individuals using RMSE

    The predictions must be sorted by score before being passed to this function.

    Args:
        ground_truth: The ground truth labels for an image, of shape (n_idv, n_bpt, 2)
        predictions: The predictions for an image, of shape (n_idv, n_bpt, 2)
        keep_assemblies: Whether to match predicted keypoints to ground truth keypoints
            while enforcing that all bodyparts for a predicted individual are matched
            to bodyparts from the same ground truth assembly. When set to False, this
            corresponds to detection RMSE score.

    Returns:
        A list containing a PotentialMatch for each predicted pose in the given
        predictions.
    """
    if not keep_assemblies:
        raise NotImplementedError()

    matches = [PotentialMatch.from_pose(pose=pred) for pred in predictions]
    matched_gt_indices = set()
    for idx, pred in enumerate(predictions):
        bpt_distances = np.linalg.norm(pred[:, :2] - ground_truth[:, :, :2], axis=-1)
        if np.all(np.isnan(bpt_distances)):
            continue

        distances = np.nanmean(bpt_distances, axis=-1)
        ind_best = np.nanargmin(distances)

        # if this gt already matched, continue
        if ind_best in matched_gt_indices:
            continue

        matched_gt_indices.add(ind_best)
        matches[idx].match(
            gt=ground_truth[ind_best],
            oks=float("nan"),  # don't compute OKS here
        )

    return matches
