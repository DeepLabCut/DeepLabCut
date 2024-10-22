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
"""TODO: Source"""
from __future__ import annotations

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    BasePredictor,
    PREDICTORS,
)


@PREDICTORS.register_module
class SimCCPredictor(BasePredictor):
    """Abstract class to generate target Simple Coordinate Classification targets

    TODO: https://github.com/open-mmlab/mmpose/blob/71ec36ebd63c475ab589afc817868e749a61491f/mmpose/codecs/simcc_label.py
    """

    def __init__(
        self,
        simcc_split_ratio: float = 2.0,
        use_dark: bool = False,
    ) -> None:
        super().__init__()
        self.simcc_split_ratio = simcc_split_ratio
        self.use_dark = use_dark

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # FIXME(niels) - process in PyTorch?
        simcc_x = outputs["x"].detach().cpu().numpy()
        simcc_y = outputs["y"].detach().cpu().numpy()
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)

        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        if self.use_dark:
            raise NotImplementedError()

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
    """Get maximum response location and value from simcc representations.

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
