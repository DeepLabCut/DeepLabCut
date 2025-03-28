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
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
from deeplabcut.pose_estimation_pytorch.data.utils import out_of_bounds_keypoints


KEYPOINT_ENCODERS = Registry("kpt_encoders", build_func=build_from_cfg)


class BaseKeypointEncoder(ABC):
    """Encodes keypoints into heatmaps

    Modified from BUCTD/data/JointsDataset
    """

    def __init__(
        self,
        num_joints: int,
        kernel_size: tuple[int, int] = (15, 15),
        img_size: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Args:
            num_joints: The number of joints to encode
            kernel_size: The Gaussian kernel size to use when blurring a heatmap
            img_size: The (height, width) of the input images
        """
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.img_size = img_size

    @property
    def num_channels(self):
        pass

    @abstractmethod
    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: the keypoints to encode
            size: the (height, width) of the heatmap in which the keypoints should
                be encoded

        Returns:
            the encoded keypoints
        """
        raise NotImplementedError

    def blur_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Applies a Gaussian blur to a heatmap

        Taken from BUCTD/data/JointsDataset, generate_heatmap

        Args:
            heatmap: the heatmap to blur (with values in [0, 1] or [0, 255])

        Returns:
            The heatmap with a Gaussian blur, such that max(heatmap) = 255
        """
        heatmap = cv2.GaussianBlur(heatmap, self.kernel_size, sigmaX=0)
        am = np.amax(heatmap)
        if am == 0:
            return heatmap
        heatmap /= am / 255
        return heatmap

    # def blur_heatmap_batch(self, heatmaps: torch.tensor) -> np.ndarray:
    #     heatmaps = TF.gaussian_blur(heatmaps.permute(0,3,1,2), self.kernel_size).permute(0,2,3,1).numpy()
    #     am = np.amax(heatmaps)
    #     if am == 0:
    #         return heatmaps
    #     heatmaps /= (am / 255)
    #     return heatmaps


@KEYPOINT_ENCODERS.register_module
class StackedKeypointEncoder(BaseKeypointEncoder):
    """Encodes keypoints into heatmaps, where each

    Modified from BUCTD/data/JointsDataset, get_stacked_condition
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def num_channels(self):
        return self.num_joints

    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: the keypoints to encode
            size: the (height, width) of the heatmap in which the keypoints should
                be encoded

        Returns:
            the encoded keypoints
        """

        batch_size, _, _ = keypoints.shape

        kpts = keypoints.copy()
        kpts[keypoints[..., 2] <= 0] = 0

        # Mark keypoints as visible, remove NaNs
        kpts[kpts[..., 2] > 0, 2] = 2
        kpts = np.nan_to_num(kpts)

        oob_mask = out_of_bounds_keypoints(kpts, self.img_size)
        if np.sum(oob_mask) > 0:
            kpts[oob_mask] = 0
        kpts = kpts.astype(int)

        zero_matrix = np.zeros((batch_size, size[0], size[1], self.num_channels))

        def _get_condition_matrix(zero_matrix, kpts):
            for i, pose in enumerate(kpts):
                x, y, vis = pose.T
                mask = vis > 0
                x_masked, y_masked, joint_inds_masked = (
                    x[mask],
                    y[mask],
                    np.arange(self.num_joints)[mask],
                )
                zero_matrix[i, y_masked - 1, x_masked - 1, joint_inds_masked] = 255
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)

        for i in range(batch_size):
            condition_heatmap = self.blur_heatmap(condition[i])
            condition[i] = condition_heatmap

        return condition


@KEYPOINT_ENCODERS.register_module
class ColoredKeypointEncoder(BaseKeypointEncoder):
    """Encodes keypoints into a given number of color channels

    Modified from BUCTD/data/JointsDataset, get_condition_image_colored
    """

    def __init__(
        self, colors: list[tuple[int, int, int]] | None = None, **kwargs
    ) -> None:
        """
        Args:
            colors: the color to use for each keypoint
        """
        super().__init__(**kwargs)
        if colors is None:
            colors = self.get_colors_from_cmap("rainbow", self.num_joints)
        self.colors = np.array(colors)

    @property
    def num_channels(self):
        return 3

    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: batch of keypoints to encode with shape (batch_size, num_joints, 2)
            size: the (height, width) of the heatmap in which the keypoints should be encoded

        Returns:
            encoded keypoints with shape (batch_size, num_joints, height, width, 3)
        """

        batch_size, num_kpts, _ = keypoints.shape

        if not num_kpts == len(self.colors):
            raise ValueError(
                f"Cannot encode the keypoints. Initialized with {len(self.colors)} "
                f"colors, but there are {num_kpts} to encode"
            )

        # kpts = keypoints.detach().numpy()
        kpts = keypoints.copy()
        kpts[keypoints[..., 2] <= 0] = 0

        # Mark keypoints as visible, remove NaNs
        kpts[kpts[..., 2] > 0, 2] = 2
        kpts = np.nan_to_num(kpts)

        oob_mask = out_of_bounds_keypoints(kpts, self.img_size)
        if np.sum(oob_mask) > 0:
            kpts[oob_mask] = 0
        kpts = kpts.astype(int)

        zero_matrix = np.zeros((batch_size, size[0], size[1], self.num_channels))

        def _get_condition_matrix(zero_matrix, kpts):
            for i, pose in enumerate(kpts):
                x, y, vis = pose.T
                mask = vis > 0
                x_masked, y_masked, colors_masked = x[mask], y[mask], self.colors[mask]
                zero_matrix[i, y_masked - 1, x_masked - 1] = colors_masked
            return zero_matrix

        def _get_condition_matrix_optim(zero_matrix, kpts):
            x, y = np.array(kpts).T
            mask = (
                (0 < x)
                & (x < zero_matrix.shape[2])
                & (0 < y)
                & (y < zero_matrix.shape[1])
            )
            colors_masked = np.repeat(
                self.colors[:, None, :], len(zero_matrix), 1
            ) * np.repeat(mask[:, :, None], 3, 2)
            kpt_indices = np.stack([x.T, y.T]).transpose(1, 2, 0)
            batch_indices = np.repeat(
                np.arange(len(zero_matrix))[:, None, None], self.num_joints, axis=1
            )
            kpt_input = np.concatenate([batch_indices, kpt_indices], dtype=int, axis=2)
            zero_matrix[
                kpt_input[..., 0], kpt_input[..., 2] - 1, kpt_input[..., 1] - 1
            ] = colors_masked.transpose(1, 0, 2)
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)
        # condition = _get_condition_matrix_optim(zero_matrix, kpts)

        for i in range(batch_size):
            condition_heatmap = self.blur_heatmap(condition[i])
            condition[i] = condition_heatmap
        # condition = self.blur_heatmap_batch(torch.from_numpy(condition))

        return condition

    def get_colors_from_cmap(self, cmap_name, num_colors):
        cmap = plt.get_cmap(cmap_name)
        colors_float = [cmap(i) for i in np.linspace(0, 256, num_colors, dtype=int)]
        colors = [
            (int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors_float
        ]
        return colors
