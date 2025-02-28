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
"""Modified SimCC target generator for the RTMPose model

Based on the official ``mmpose`` SimCC codec and RTMCC head implementation. For more
information, see <https://github.com/open-mmlab/mmpose>.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    BaseGenerator,
    TARGET_GENERATORS,
)


@TARGET_GENERATORS.register_module
class SimCCGenerator(BaseGenerator):
    """Class used generate targets from RTMPose head outputs

    The RTMPose model uses coordinate classification for pose estimation. For more
    information, see "SimCC: a Simple Coordinate Classification Perspective for Human
    Pose Estimation" (<https://arxiv.org/pdf/2107.03332>) and "RTMPose: Real-Time
    Multi-Person Pose Estimation based on MMPose" (<https://arxiv.org/pdf/2303.07399>).

    Args:
        input_size: The size of images given to the pose estimation model.
        smoothing_type: Smoothing strategy ("gaussian" or "standard")
        sigma: The sigma value in the Gaussian SimCC label. If a single value, used for
            both x and y. If two values, the sigmas for (x, y).
        simcc_split_ratio: The split ratio of pixels, as described in SimCC.
        label_smooth_weight: Label Smoothing weight.
        normalize: Normalize the heatmaps before returning.
        **kwargs,
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        smoothing_type: str = "gaussian",
        sigma: float | int | tuple[float, ...] = 6.0,
        simcc_split_ratio: float = 2.0,
        label_smooth_weight: float = 0.0,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma])
        else:
            self.sigma = np.array(sigma)

        if self.smoothing_type not in {"gaussian", "standard"}:
            raise ValueError(
                f"{self.__class__.__name__} got invalid `smoothing_type` value"
                f"{self.smoothing_type}. Should be one of "
                '{"gaussian", "standard"}'
            )

        if self.smoothing_type == "gaussian" and self.label_smooth_weight > 0:
            raise ValueError(
                "Attribute `label_smooth_weight` is only " "used for `standard` mode."
            )

        if self.label_smooth_weight < 0.0 or self.label_smooth_weight > 1.0:
            raise ValueError("`label_smooth_weight` should be in range [0, 1]")

        if self.smoothing_type == "gaussian":
            self.generator = self._generate_gaussian
        elif self.smoothing_type == "standard":
            self.generator = self._generate_standard
        else:
            raise ValueError(
                f"{self.__class__.__name__} got invalid `smoothing_type` value"
                f"{self.smoothing_type}. Should be one of "
                '{"gaussian", "standard"}'
            )

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor], labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        device = outputs["x"].device
        keypoints = labels[self.label_keypoint_key].cpu().numpy()
        batch_size = len(keypoints)

        if len(keypoints.shape) == 3:  # for single animal: add individual dimension
            keypoints = keypoints.reshape((batch_size, 1, *keypoints.shape[1:]))

        xs, ys, ws = [], [], []
        for batch_keypoints in keypoints:
            keypoints = batch_keypoints[:, :, :2]
            keypoints_visible = batch_keypoints[:, :, 2]
            x_labels, y_labels, weights = self.generator(keypoints, keypoints_visible)
            xs.append(x_labels)
            ys.append(y_labels)
            ws.append(weights)

        x_labels = np.stack(xs)
        y_labels = np.stack(ys)
        weights = np.stack(ws)
        return dict(
            x=dict(
                target=torch.tensor(x_labels, device=device),
                weights=torch.tensor(weights, device=device),
            ),
            y=dict(
                target=torch.tensor(y_labels, device=device),
                weights=torch.tensor(weights, device=device),
            ),
        )

    def _generate_standard(
        self, keypoints: np.ndarray, keypoints_visible: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Standard Label Smoothing.

        Labels will be one-hot vectors if self.label_smooth_weight==0.0
        """
        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible
        )

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabeled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get center coordinates
            mu_x, mu_y = keypoints_split[n, k].astype(np.int64)

            # detect abnormal coords and assign the weight 0
            if mu_x >= W or mu_y >= H or mu_x < 0 or mu_y < 0:
                keypoint_weights[n, k] = 0
                continue

            if self.label_smooth_weight > 0:
                target_x[n, k] = self.label_smooth_weight / (W - 1)
                target_y[n, k] = self.label_smooth_weight / (H - 1)

            target_x[n, k, mu_x] = 1.0 - self.label_smooth_weight
            target_y[n, k, mu_y] = 1.0 - self.label_smooth_weight

        return target_x, target_y, keypoint_weights

    def _map_coordinates(
        self, keypoints: np.ndarray, keypoints_visible: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mapping keypoint coordinates into SimCC space"""
        keypoints_split = keypoints.copy()
        # set non-visible keypoints to 0; deals with NaNs
        keypoints_split[keypoints_visible <= 0] = 0
        keypoints_split = np.around(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = (keypoints_visible > 0).astype(keypoints_split.dtype)
        return keypoints_split, keypoint_weights

    def _generate_gaussian(
        self, keypoints: np.ndarray, keypoints_visible: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label Smoothing"""
        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible
        )

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            # skip unlabeled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            target_x[n, k] = np.exp(-((x - mu_x) ** 2) / (2 * self.sigma[0] ** 2))
            target_y[n, k] = np.exp(-((y - mu_y) ** 2) / (2 * self.sigma[1] ** 2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]

        return target_x, target_y, keypoint_weights
