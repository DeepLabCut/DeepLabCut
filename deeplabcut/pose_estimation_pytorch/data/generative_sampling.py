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
"""A file containing code to perform generative sampling of keypoints for CTD

This code comes from PoseFix (see https://arxiv.org/pdf/1812.03595.pdf), and was then
adapted for BUCTD (github.com/amathislab/BUCTD/blob/main/lib/dataset/pose_synthesis.py,
see `synthesize_pose_fish(...)`).
They say:
> ... synthesized poses need to be diverse and realistic. To satisfy these properties,
> we generate synthesized poses randomly based on the error distributions of real poses
> as described in [24]. The distributions include the frequency of each pose error
> (i.e., jitter, inversion, swap, and miss) according to the joint type, number of
> visible keypoints, and overlap in the input image.
> ...
> Types of Keypoints:
> Good. Good status is defined as a very small displacement from the GT keypoint.
> Jitter. Jitter error is defined as a small displacement from the GT keypoint.
> Inversion. Inversion error occurs when a pose estimation model is confused between
>   semantically similar parts that belong to the same instance.
> Swap. Swap error represents a confusion between the same or similar parts which belong
>   to different persons.
> Miss. Miss error represents a large displacement from the GT keypoint position.

In BUCTD and their adaptation to the maDLC fish dataset, they set:
    if cfg.DATASET.DATASET == 'coco':
        kps_symmetry = [(1, 2), (3, 4), (5, 6), ...]
        kps_sigmas = np.array([.26, .25, .25, ...]) / 10.0
    elif cfg.DATASET.DATASET == 'crowdpose':
        kps_sigmas = np.array([.79, .79, .72, ...])/10.0
        kps_symmetry= [(0, 1), (2, 3), (4, 5), ...]  # l/r shoulder, l/r elbow,  wrist,
    else:
        kps_symmetry = []
        kps_sigmas = np.array([1.] * num_kpts)/10.0
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

import cv2
import numpy as np

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg


KEYPOINT_ENCODERS = Registry("detectors", build_func=build_from_cfg)


class BaseKeypointEncoder(ABC):
    """Encodes keypoints into heatmaps

    Modified from BUCTD/data/JointsDataset
    """

    def __init__(self, kernel_size: tuple[int, int] = (15, 15)) -> None:
        """
        Args:
            kernel_size: the Gaussian kernel size to use when blurring a heatmap
        """
        self.kernel_size = kernel_size

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

    def __init__(self, colors: list[float], **kwargs) -> None:
        """
        Args:
            colors: the color to use for each keypoint
        """
        super().__init__(**kwargs)
        self.colors = colors

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


class GenerativeSampler:
    """Performs generative sampling of keypoints for CTD model training"""

    def __init__(
        self,
        num_keypoints: int,
        keypoint_sigmas: float | list[float] = 0.1,
        keypoints_symmetry: list[tuple[int, int]] | None = None
    ):
        """
        Args:
            num_keypoints: the number of keypoints per individual
            keypoint_sigmas: the sigma for each keypoint
            keypoints_symmetry: indices of keypoints that are symmetric (e.g., left and
                right eye)
        """
        if isinstance(keypoint_sigmas, float):
            keypoint_sigmas = num_keypoints * [keypoint_sigmas]
        if keypoints_symmetry is None:
            keypoints_symmetry = keypoints_symmetry

        self.keypoint_sigmas = np.array(keypoint_sigmas)
        self.keypoints_symmetry = keypoints_symmetry
        self.num_keypoints = num_keypoints

    def __call__(
        self,
        keypoints: np.ndarray,
        near_keypoints: np.ndarray,
        area: float,  # ??
        #num_overlap: int,  # ??
    ) -> np.ndarray:
        """Samples keypoints

        PoseFix uses conditional keypoints (estimated by a bottom-up model) when ground
        truth keypoints are not available. For simplicity, we omit that. See
        https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/gen_batch.py#L76

        Args:
            keypoints: (num_keypoints, x-y-visibility) the ground truth keypoints
            near_keypoints: (num_other_individuals, num_keypoints, x-y-visibility) joints
                from other individuals near this one, for which keypoints might be swapped
            area: the total area of the bounding box surrounding the keypoints
            num_overlap:

        Returns:
            the generative sampled keypoints, of shape (num_keypoints, x-y-visibility)
        """
        if not keypoints.shape[0] == self.num_keypoints:
            raise ValueError(f"Expected {self.num_keypoints} kpts, had {keypoints}")

        ks_10_dist = self.get_distance_wrt_keypoint_sim(0.10, area)
        ks_50_dist = self.get_distance_wrt_keypoint_sim(0.50, area)
        ks_85_dist = self.get_distance_wrt_keypoint_sim(0.85, area)

        synth_joints = keypoints.copy()
        # FIXME: In the original codebase, if some keypoints are not annotated then they
        #  use the predictions made by a pose model. This is complex to integrate into
        #  the current codebase (where is the prediction file saved? how do we load
        #  predictions? which model?) so we ignore it for now
        # for j in range(self.num_keypoints):
        #     # in case of not annotated joints, use other models`s result and add noise
        #     if joints[j, 2] == 0:
        #         synth_joints[j] = estimated_joints[j]

        num_valid_joint = np.sum(keypoints[:, 2] > 0)

        N = 500  # TODO: do not know how this is set
        for j in range(self.num_keypoints):

            # source keypoint position candidates to generate error on that (gt, swap, inv, swap+inv)
            coord_list = []
            # on top of gt
            gt_coord = np.expand_dims(synth_joints[j, :2], 0)
            coord_list.append(gt_coord)
            # on top of swap gt
            swap_coord = near_keypoints[near_keypoints[:, j, 2] > 0, j, :2]
            coord_list.append(swap_coord)

            # on top of inv gt, swap inv gt
            # FIXME: In the original codebase, they only swap symmetric keypoints. As
            #  we don't always have symmetries for keypoints in DeepLabCut, we swap any
            #  keypoint with any other keypoint by randomly selecting keypoints to swap
            kps_symmetry = self.keypoints_symmetry
            pair_exist = False
            for (q, w) in kps_symmetry:
                if j == q or j == w:
                    if j == q:
                        pair_idx = w
                    else:
                        pair_idx = q
                    pair_exist = True
            if pair_exist and (keypoints[pair_idx, 2] > 0):
                inv_coord = np.expand_dims(synth_joints[pair_idx, :2], 0)
                coord_list.append(inv_coord)
            else:
                coord_list.append(np.empty([0, 2]))

            if pair_exist:
                swap_inv_coord = near_keypoints[near_keypoints[:, pair_idx, 2] > 0, pair_idx, :2]
                coord_list.append(swap_inv_coord)
            else:
                coord_list.append(np.empty([0, 2]))

            # shape (s, 2)
            tot_coord_list = np.concatenate(coord_list)

            assert len(coord_list) == 4

            # jitter error
            synth_jitter = np.zeros(3)
            
            # if num_valid_joint <= 4:
            #     jitter_prob = 0.20
            # else:
            #     jitter_prob = 0.15
            jitter_prob = 0.16

            angle = np.random.uniform(0, 2 * math.pi, [N])
            r = np.random.uniform(ks_85_dist[j], ks_50_dist[j], [N])
            jitter_idx = 0  # gt
            x = tot_coord_list[jitter_idx][0] + r * np.cos(angle)
            y = tot_coord_list[jitter_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == jitter_idx:
                    continue
                dist_mask = np.logical_and(
                    dist_mask, np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r
                )
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_jitter[0] = x[rand_idx]
                synth_jitter[1] = y[rand_idx]
                synth_jitter[2] = 1

            # miss error
            synth_miss = np.zeros(3)
            
            # if num_valid_joint <= 2:
            #     miss_prob = 0.20
            # elif num_valid_joint <= 4:
            #     miss_prob = 0.13
            # else:
            #     miss_prob = 0.05
            miss_prob = 0.10

            miss_pt_list = []
            for miss_idx in range(len(tot_coord_list)):
                angle = np.random.uniform(0, 2 * math.pi, [4 * N])
                r = np.random.uniform(ks_50_dist[j], ks_10_dist[j], [4 * N])
                x = tot_coord_list[miss_idx][0] + r * np.cos(angle)
                y = tot_coord_list[miss_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == miss_idx:
                        continue
                    dist_mask = np.logical_and(
                        dist_mask,
                        np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > ks_50_dist[
                            j]
                    )
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    if miss_idx == 0:
                        coord = np.transpose(np.vstack([x, y]), [1, 0])
                        miss_pt_list.append(coord)
                    else:
                        rand_idx = np.random.choice(range(len(x)), size=len(x) // 4)
                        x = np.take(x, rand_idx)
                        y = np.take(y, rand_idx)
                        coord = np.transpose(np.vstack([x, y]), [1, 0])
                        miss_pt_list.append(coord)
            if len(miss_pt_list) > 0:
                miss_pt_list = np.concatenate(miss_pt_list, axis=0).reshape(-1, 2)
                rand_idx = random.randrange(0, len(miss_pt_list))
                synth_miss[0] = miss_pt_list[rand_idx][0]
                synth_miss[1] = miss_pt_list[rand_idx][1]
                synth_miss[2] = 1

            # inversion prob
            synth_inv = np.zeros(3)
            inv_prob = 0.03
            if pair_exist and keypoints[pair_idx, 2] > 0:
                angle = np.random.uniform(0, 2 * math.pi, [N])
                r = np.random.uniform(0, ks_50_dist[j], [N])
                inv_idx = (len(coord_list[0]) + len(coord_list[1]))
                x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
                y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == inv_idx:
                        continue
                    dist_mask = np.logical_and(
                        dist_mask, np.sqrt(
                            (tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2
                        ) > r
                    )
                x = x[dist_mask].reshape(-1)
                y = y[dist_mask].reshape(-1)
                if len(x) > 0:
                    rand_idx = random.randrange(0, len(x))
                    synth_inv[0] = x[rand_idx]
                    synth_inv[1] = y[rand_idx]
                    synth_inv[2] = 1

            # swap prob
            synth_swap = np.zeros(3)
            swap_exist = (len(coord_list[1]) > 0) or (len(coord_list[3]) > 0)
            
            # if (num_valid_joint <= 4 and num_overlap > 0) or (num_valid_joint <= 5 and num_overlap >= 1):
            #     swap_prob = 0.10
            # else:
            #     swap_prob = 0.04
            swap_prob = 0.08
            
            if swap_exist:
                swap_pt_list = []
                for swap_idx in range(len(tot_coord_list)):
                    if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(coord_list[1]):
                        continue
                    angle = np.random.uniform(0, 2 * math.pi, [N])
                    r = np.random.uniform(0, ks_50_dist[j], [N])
                    x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                    y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                    dist_mask = True
                    for i in range(len(tot_coord_list)):
                        if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                            dist_mask = np.logical_and(
                                dist_mask, np.sqrt(
                                    (tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2
                                ) > r
                            )
                    x = x[dist_mask].reshape(-1)
                    y = y[dist_mask].reshape(-1)
                    if len(x) > 0:
                        coord = np.transpose(np.vstack([x, y]), [1, 0])
                        swap_pt_list.append(coord)
                if len(swap_pt_list) > 0:
                    swap_pt_list = np.concatenate(swap_pt_list, axis=0).reshape(-1, 2)
                    rand_idx = random.randrange(0, len(swap_pt_list))
                    synth_swap[0] = swap_pt_list[rand_idx][0]
                    synth_swap[1] = swap_pt_list[rand_idx][1]
                    synth_swap[2] = 1

            # good prob
            synth_good = np.zeros(3)
            good_prob = 1 - (jitter_prob + miss_prob + inv_prob + swap_prob)
            assert good_prob >= 0
            angle = np.random.uniform(0, 2 * math.pi, [N // 4])
            r = np.random.uniform(0, ks_85_dist[j], [N // 4])
            good_idx = 0  # gt
            x = tot_coord_list[good_idx][0] + r * np.cos(angle)
            y = tot_coord_list[good_idx][1] + r * np.sin(angle)
            dist_mask = True
            for i in range(len(tot_coord_list)):
                if i == good_idx:
                    continue
                dist_mask = np.logical_and(
                    dist_mask, np.sqrt((tot_coord_list[i][0] - x) ** 2 + (tot_coord_list[i][1] - y) ** 2) > r
                )
            x = x[dist_mask].reshape(-1)
            y = y[dist_mask].reshape(-1)
            if len(x) > 0:
                rand_idx = random.randrange(0, len(x))
                synth_good[0] = x[rand_idx]
                synth_good[1] = y[rand_idx]
                synth_good[2] = 1

            if synth_jitter[2] == 0:
                jitter_prob = 0
            if synth_inv[2] == 0:
                inv_prob = 0
            if synth_swap[2] == 0:
                swap_prob = 0
            if synth_miss[2] == 0:
                miss_prob = 0
            if synth_good[2] == 0:
                good_prob = 0

            normalizer = jitter_prob + miss_prob + inv_prob + swap_prob + good_prob
            if normalizer == 0:
                synth_joints[j] = 0
                continue

            jitter_prob = jitter_prob / normalizer
            miss_prob = miss_prob / normalizer
            inv_prob = inv_prob / normalizer
            swap_prob = swap_prob / normalizer
            good_prob = good_prob / normalizer

            prob_list = [jitter_prob, miss_prob, inv_prob, swap_prob, good_prob]
            synth_list = [synth_jitter, synth_miss, synth_inv, synth_swap, synth_good]
            sampled_idx = np.random.choice(5, 1, p=prob_list)[0]
            synth_joints[j] = synth_list[sampled_idx]
            synth_joints[j, 2] = 0

        return synth_joints

    def get_distance_wrt_keypoint_sim(self, ks: float, area: float) -> np.ndarray:
        """
        Args:
            ks: the desired keypoint similarity
            area: the area of the bounding box for the individual

        Returns:
            For each bodypart, the L2 distance for which the keypoint similarity is
            equal to ks
        """
        return np.sqrt(-2 * area * ((self.keypoint_sigmas * 2) ** 2) * np.log(ks))
