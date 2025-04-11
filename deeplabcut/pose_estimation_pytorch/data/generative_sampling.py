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
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class GenSamplingConfig:
    """Configuration for CTD models.

    Args:
        bbox_margin: The margin added around conditional keypoints
        keypoint_sigmas: The sigma for each keypoint.
        keypoints_symmetry: Indices of symmetric keypoints (e.g. left/right eye)
        jitter_prob: The probability of applying jitter. Jitter error is defined as
            a small displacement from the GT keypoint.
        swap_prob: The probability of applying a swap error. Swap error represents
            a confusion between the same or similar parts which belong to different
            persons.
        inv_prob: The probability of applying an inversion error. Inversion error
            occurs when a pose estimation model is confused between semantically
            similar parts that belong to the same instance.
        miss_prob: The probability of applying a miss error. Miss error represents a
            large displacement from the GT keypoint position.
    """
    bbox_margin: int
    keypoint_sigmas: float | list[float] = 0.1
    keypoints_symmetry: list[tuple[int, int]] | None = None
    jitter_prob: float = 0.16
    swap_prob: float = 0.08
    inv_prob: float = 0.03
    miss_prob: float = 0.10

    def to_dict(self) -> dict:
        return {
            "keypoint_sigmas": self.keypoint_sigmas,
            "keypoints_symmetry": self.keypoints_symmetry,
            "jitter_prob": self.jitter_prob,
            "swap_prob": self.swap_prob,
            "inv_prob": self.inv_prob,
            "miss_prob": self.miss_prob,
        }


class GenerativeSampler:
    """Performs generative sampling of keypoints for CTD model training"""

    def __init__(
        self,
        num_keypoints: int,
        keypoint_sigmas: float | list[float] = 0.1,
        keypoints_symmetry: list[tuple[int, int]] | None = None,
        jitter_prob: float = 0.16,
        swap_prob: float = 0.08,
        inv_prob: float = 0.03,
        miss_prob: float = 0.10,
    ):
        """
        Args:
            num_keypoints: the number of keypoints per individual
            keypoint_sigmas: the sigma for each keypoint
            keypoints_symmetry: indices of keypoints that are symmetric (e.g., left and
                right eye)
            jitter_prob: The probability of applying jitter. Jitter error is defined as
                a small displacement from the GT keypoint.
            swap_prob: The probability of applying a swap error. Swap error represents
                a confusion between the same or similar parts which belong to different
                persons.
            inv_prob: The probability of applying an inversion error. Inversion error
                occurs when a pose estimation model is confused between semantically
                similar parts that belong to the same instance.
            miss_prob: The probability of applying a miss error. Miss error represents a
                large displacement from the GT keypoint position.
        """
        if isinstance(keypoint_sigmas, float):
            keypoint_sigmas = num_keypoints * [keypoint_sigmas]

        self.keypoint_sigmas = np.array(keypoint_sigmas)
        self.keypoints_symmetry = keypoints_symmetry
        self.num_keypoints = num_keypoints
        self.jitter_prob = jitter_prob
        self.swap_prob = swap_prob
        self.inv_prob = inv_prob
        self.miss_prob = miss_prob

    def __call__(
        self,
        keypoints: np.ndarray,
        near_keypoints: np.ndarray,
        area: float,
        image_size: tuple[int, int],
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
        #     if keypoints[j, 2] == 0:
        #         synth_joints[j] = estimated_joints[j]

        # num_valid_joint = np.sum(keypoints[:, 2] > 0)

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
            if self.keypoints_symmetry is None or len(self.keypoints_symmetry) == 0:
                # randomly sample keypoint pairs to swap
                kps_symmetry = np.random.choice(
                    list(range(self.num_keypoints)),
                    size=(self.num_keypoints // 2, 2),
                    replace=False,
                )
            else:
                kps_symmetry = self.keypoints_symmetry

            pair_idx = None
            for q, w in kps_symmetry:
                if j == q or j == w:
                    if j == q:
                        pair_idx = w
                    else:
                        pair_idx = q

            if pair_idx is not None and (keypoints[pair_idx, 2] > 0):
                inv_coord = np.expand_dims(synth_joints[pair_idx, :2], 0)
                coord_list.append(inv_coord)
            else:
                coord_list.append(np.empty([0, 2]))

            if pair_idx is not None:
                swap_inv_coord = near_keypoints[
                    near_keypoints[:, pair_idx, 2] > 0, pair_idx, :2
                ]
                coord_list.append(swap_inv_coord)
            else:
                coord_list.append(np.empty([0, 2]))

            # shape (s, 2)
            tot_coord_list = np.concatenate(coord_list)

            assert len(coord_list) == 4

            # jitter error
            synth_jitter = np.zeros(3)
            jitter_prob = self.jitter_prob

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
                    dist_mask,
                    np.sqrt(
                        (tot_coord_list[i][0] - x) ** 2
                        + (tot_coord_list[i][1] - y) ** 2
                    )
                    > r,
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
            miss_prob = self.miss_prob

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
                        np.sqrt(
                            (tot_coord_list[i][0] - x) ** 2
                            + (tot_coord_list[i][1] - y) ** 2
                        )
                        > ks_50_dist[j],
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
            inv_prob = self.inv_prob
            if pair_idx is not None and keypoints[pair_idx, 2] > 0:
                angle = np.random.uniform(0, 2 * math.pi, [N])
                r = np.random.uniform(0, ks_50_dist[j], [N])
                inv_idx = len(coord_list[0]) + len(coord_list[1])
                x = tot_coord_list[inv_idx][0] + r * np.cos(angle)
                y = tot_coord_list[inv_idx][1] + r * np.sin(angle)
                dist_mask = True
                for i in range(len(tot_coord_list)):
                    if i == inv_idx:
                        continue
                    dist_mask = np.logical_and(
                        dist_mask,
                        np.sqrt(
                            (tot_coord_list[i][0] - x) ** 2
                            + (tot_coord_list[i][1] - y) ** 2
                        )
                        > r,
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
            swap_prob = self.swap_prob

            if swap_exist:
                swap_pt_list = []
                for swap_idx in range(len(tot_coord_list)):
                    if swap_idx == 0 or swap_idx == len(coord_list[0]) + len(
                        coord_list[1]
                    ):
                        continue
                    angle = np.random.uniform(0, 2 * math.pi, [N])
                    r = np.random.uniform(0, ks_50_dist[j], [N])
                    x = tot_coord_list[swap_idx][0] + r * np.cos(angle)
                    y = tot_coord_list[swap_idx][1] + r * np.sin(angle)
                    dist_mask = True
                    for i in range(len(tot_coord_list)):
                        if i == 0 or i == len(coord_list[0]) + len(coord_list[1]):
                            dist_mask = np.logical_and(
                                dist_mask,
                                np.sqrt(
                                    (tot_coord_list[i][0] - x) ** 2
                                    + (tot_coord_list[i][1] - y) ** 2
                                )
                                > r,
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
                    dist_mask,
                    np.sqrt(
                        (tot_coord_list[i][0] - x) ** 2
                        + (tot_coord_list[i][1] - y) ** 2
                    )
                    > r,
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
            synth_joints[j, 2] = 2

        nan_mask = np.isnan(synth_joints).any(axis=1)
        synth_joints[nan_mask, 2] = 0
        np.clip(synth_joints[:, 0], 0, image_size[1], out=synth_joints[:, 0])
        np.clip(synth_joints[:, 1], 0, image_size[0], out=synth_joints[:, 1])
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
