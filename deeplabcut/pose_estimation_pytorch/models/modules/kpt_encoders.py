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
import matplotlib.pyplot as plt

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


KEYPOINT_ENCODERS = Registry("kpt_encoders", build_func=build_from_cfg)


class BaseKeypointEncoder(ABC):
    """Encodes keypoints into heatmaps

    Modified from BUCTD/data/JointsDataset
    """

    def __init__(self, num_joints, kernel_size: tuple[int, int] = (15, 15)) -> None:
        """
        Args:
            kernel_size: the Gaussian kernel size to use when blurring a heatmap
        """
        self.kernel_size = kernel_size
        self.num_joints = num_joints

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
        heatmap /= (am / 255)
        return heatmap


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
        kpts = np.array(keypoints).astype(int)  # .reshape(-1, 2).astype(int)
        zero_matrix = np.zeros(size)

        def _get_condition_matrix(zero_matrix_, kpt_):
            if 0 < kpt_[0] < size[1] and 0 < kpt_[1] < size[0]:
                zero_matrix_[kpt_[1] - 1][kpt_[0] - 1] = 255
            return zero_matrix_

        condition_heatmap_list = []
        for i, kpt in enumerate(kpts):
            condition = _get_condition_matrix(zero_matrix, kpt)
            condition_heatmap = self.blur_heatmap(condition)
            condition_heatmap_list.append(condition_heatmap)
            zero_matrix = np.zeros(size)

            # ### debug: visualization -> check conditions
            # condition_heatmap = np.expand_dims(condition_heatmap, axis=0)
            # condition = np.repeat(condition_heatmap, 3, axis=0)
            # print("condition", condition.shape)
            # condition = np.transpose(condition, (1, 2, 0))
            # cv2.imwrite(f'/media/data/mu/test/cond_{i}.jpg', condition+image)
            # cv2.imwrite(f'/media/data/mu/test/image.jpg', image)

        condition_heatmap_list = np.moveaxis(np.array(condition_heatmap_list), 0, -1)
        return condition_heatmap_list


@KEYPOINT_ENCODERS.register_module
class ColoredKeypointEncoder(BaseKeypointEncoder):
    """Encodes keypoints into a given number of color channels

    Modified from BUCTD/data/JointsDataset, get_condition_image_colored
    """

    def __init__(self, **kwargs) -> None:
        """
        Args:
            colors: the color to use for each keypoint
        """
        super().__init__(**kwargs)
        self.colors = self.get_colors_from_cmap('rainbow', self.num_joints)

    @property
    def num_channels(self):
        return 3

    def __call__(self, keypoints: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            keypoints: the keypoints to encode
            size: the (height, width) of the heatmap in which the keypoints should
                be encoded

        Returns:
            the encoded keypoints
        """
        if not len(keypoints) == len(self.colors):
            raise ValueError(
                f"Cannot encode the keypoints. Initialized with {len(self.colors)} "
                f"colors, but there are {len(keypoints)} to encode"
            )

        kpts = np.array(keypoints).astype(int)  # .reshape(-1, 2).astype(int)
        zero_matrix = np.zeros(size)

        def _get_condition_matrix(zero_matrix, kpts):
            for color, kpt in zip(self.colors, kpts):
                if 0 < kpt[0] < size[1] and 0 < kpt[1] < size[0]:
                    zero_matrix[kpt[1] - 1][kpt[0] - 1] = color
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)
        condition_heatmap = self.blur_heatmap(condition)
        return condition_heatmap

    def get_colors_from_cmap(self, cmap_name, num_colors):
        cmap = plt.get_cmap(cmap_name)
        colors_float = [cmap(i) for i in range(0, 256, 256 // num_colors)]
        colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_float]
        return colors
