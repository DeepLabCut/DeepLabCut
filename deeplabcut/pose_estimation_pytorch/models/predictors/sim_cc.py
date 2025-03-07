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
"""SimCC predictor for the RTMPose model

Based on the official ``mmpose`` SimCC codec and RTMCC head implementation. For more
information, see <https://github.com/open-mmlab/mmpose>.
"""
from __future__ import annotations

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    BasePredictor,
    PREDICTORS,
)


@PREDICTORS.register_module
class SimCCPredictor(BasePredictor):
    """Class used to make pose predictions from RTMPose head outputs

    The RTMPose model uses coordinate classification for pose estimation. For more
    information, see "SimCC: a Simple Coordinate Classification Perspective for Human
    Pose Estimation" (<https://arxiv.org/pdf/2107.03332>) and "RTMPose: Real-Time
    Multi-Person Pose Estimation based on MMPose" (<https://arxiv.org/pdf/2303.07399>).

    Args:
        simcc_split_ratio: The split ratio of pixels, as described in SimCC.
        apply_softmax: Whether to apply softmax on the scores.
        normalize_outputs: Whether to normalize the outputs before predicting maximums.
    """

    def __init__(
        self,
        simcc_split_ratio: float = 2.0,
        apply_softmax: bool = False,
        normalize_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.simcc_split_ratio = simcc_split_ratio
        self.apply_softmax = apply_softmax
        self.normalize_outputs = normalize_outputs

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        x, y = outputs["x"].detach(), outputs["y"].detach()
        if self.normalize_outputs:
            x = get_simcc_normalized(x)
            y = get_simcc_normalized(y)

        keypoints, scores = get_simcc_maximum(
            x.cpu().numpy(), y.cpu().numpy(), self.apply_softmax
        )

        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        keypoints /= self.simcc_split_ratio
        scores = scores.reshape((*scores.shape, -1))
        keypoints_with_score = np.concatenate([keypoints, scores], axis=-1)
        keypoints_with_score = torch.tensor(keypoints_with_score).unsqueeze(1)
        return dict(poses=keypoints_with_score)


def get_simcc_maximum(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    apply_softmax: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from SimCC representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), "simcc_x should be numpy.ndarray"
    assert isinstance(simcc_y, np.ndarray), "simcc_y should be numpy.ndarray"
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, f"Invalid shape {simcc_x.shape}"
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, f"Invalid shape {simcc_y.shape}"
    assert simcc_x.ndim == simcc_y.ndim, f"{simcc_x.shape} != {simcc_y.shape}"

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals


def get_simcc_normalized(pred: torch.Tensor) -> torch.Tensor:
    """Normalize the predicted SimCC.

    See:
    github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/utils/post_processing.py#L12

    Args:
        pred: The predicted output.

    Returns:
        The normalized output.
    """
    b, k, _ = pred.shape
    pred = pred.clamp(min=0)

    # Compute the binary mask
    mask = (pred.amax(dim=-1) > 1).reshape(b, k, 1)

    # Normalize the tensor using the maximum value
    norm = (pred / pred.amax(dim=-1).reshape(b, k, 1))

    # return the normalized tensor
    return torch.where(mask, norm, pred)
