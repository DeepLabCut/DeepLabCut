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

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    BaseGenerator,
    TARGET_GENERATORS,
)


@TARGET_GENERATORS.register_module
class DEKRGenerator(BaseGenerator):
    """
    Generate ground truth target for DEKR model training based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang, CVPR 2021
    Code based on:
        https://github.com/HRNet/DEKR
    """

    def __init__(
        self, num_joints: int, pos_dist_thresh: int, bg_weight: float = 0.1, **kwargs
    ):
        """
        Args:
            num_joints: number of keypoints
            pos_dist_thresh: 3*std of the gaussian
            bg_weight:background weight. Defaults to 0.1.
        """
        super().__init__(**kwargs)

        self.num_joints = num_joints
        self.num_heatmaps = self.num_joints + 1
        self.pos_dist_thresh = pos_dist_thresh
        self.bg_weight = bg_weight

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor], labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Given the annotations and predictions of your keypoints, this function returns the targets,
        a dictionary containing the heatmaps, locref_maps and locref_masks.
        Args:
            stride: the stride of the model
            outputs: output of each model head
            labels: the labels for the inputs (each tensor should have shape (b, ...))

        Returns:
            The targets for the DEKR heatmap and offset heads:
                {
                    "heatmap": {
                        "target": heatmaps,
                        "weights":  heatmap_weights,
                    },
                    "offset": {
                        "target": offset_map,
                        "weights": offset_weights,
                    }
                }

        Examples:
            input:
                labels = {"keypoints":torch.randint(1,min(image_size),(batch_size, num_animals, num_joints, 2))}
                prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
                image_size = (256, 256)
            output:
                targets = {
                    "heatmap": {"target": heatmaps, "weights":  heatmap_weights},
                    "offset": {"target": offset_map, "weights": offset_masks}
                }
        """
        stride_y, stride_x = stride, stride
        batch_size, _, output_h, output_w = outputs["heatmap"].shape
        coords = labels[self.label_keypoint_key].cpu().numpy()
        area = labels["area"].cpu().numpy()

        assert (
            self.num_joints + 1 == coords.shape[2]
        ), f"the number of joints should be {coords.shape}"

        # TODO make it possible to differentiate between center sigma and other sigmas
        scale = max(1 / stride_x, 1 / stride_y)
        sgm, ct_sgm = (self.pos_dist_thresh / 2) * scale, self.pos_dist_thresh * scale
        radius = self.pos_dist_thresh * scale

        heatmap_shape = batch_size, self.num_heatmaps, output_h, output_w
        heatmaps = np.zeros(heatmap_shape, dtype=np.float32)
        heatmap_weights = 2 * np.ones(heatmap_shape, dtype=np.float32)

        offset_shape = batch_size, self.num_joints * 2, output_h, output_w
        offset_map = np.zeros(offset_shape, dtype=np.float32)
        weight_map = np.zeros(offset_shape, dtype=np.float32)

        area_map = np.zeros((batch_size, output_h, output_w), dtype=np.float32)
        for b in range(batch_size):
            for person_id, p in enumerate(coords[b]):
                idx_center = len(p) - 1
                ct_x = int(p[-1, 0])
                ct_y = int(p[-1, 1])

                ct_x_sm = (ct_x - stride_x / 2) / stride_x
                ct_y_sm = (ct_y - stride_y / 2) / stride_y
                for idx, pt in enumerate(p):
                    if pt[-1] == -1:
                        # full gradient masking
                        heatmap_weights[b, idx] = 0.0
                        continue
                    elif pt[-1] <= 0:
                        continue

                    if idx == idx_center:
                        sigma = ct_sgm
                    else:
                        sigma = sgm

                    x, y = pt[0], pt[1]
                    x_sm, y_sm = (
                        (x - stride_x / 2) / stride_x,
                        (y - stride_y / 2) / stride_y,
                    )

                    if x_sm < 0 or y_sm < 0 or x_sm >= output_w or y_sm >= output_h:
                        continue

                    # HEATMAP COMPUTATION
                    ul = (
                        int(np.floor(x_sm - 3 * sigma - 1)),
                        int(np.floor(y_sm - 3 * sigma - 1)),
                    )
                    br = (
                        int(np.ceil(x_sm + 3 * sigma + 2)),
                        int(np.ceil(y_sm + 3 * sigma + 2)),
                    )

                    cc, dd = max(0, ul[0]), min(br[0], output_w)
                    aa, bb = max(0, ul[1]), min(br[1], output_h)

                    joint_rg = np.zeros((bb - aa, dd - cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = dekr_heatmap_val(
                                sigma, sx, sy, x_sm, y_sm
                            )

                    heatmaps[b, idx, aa:bb, cc:dd] = np.maximum(
                        heatmaps[b, idx, aa:bb, cc:dd], joint_rg
                    )
                    heatmap_weights[b, idx, aa:bb, cc:dd] = 1.0

                    # OFFSET COMPUTATION
                    if idx != idx_center:
                        start_x = max(int(ct_x_sm - radius), 0)
                        start_y = max(int(ct_y_sm - radius), 0)
                        end_x = min(int(ct_x_sm + radius), output_w)
                        end_y = min(int(ct_y_sm + radius), output_h)

                        for pos_x in range(start_x, end_x):
                            for pos_y in range(start_y, end_y):
                                offset_x = pos_x - x_sm
                                offset_y = pos_y - y_sm
                                if (
                                    offset_map[b, idx * 2, pos_y, pos_x] != 0
                                    or offset_map[b, idx * 2 + 1, pos_y, pos_x] != 0
                                ):
                                    if area_map[b, pos_y, pos_x] < area[b, person_id]:
                                        continue
                                offset_map[b, idx * 2, pos_y, pos_x] = offset_x
                                offset_map[b, idx * 2 + 1, pos_y, pos_x] = offset_y
                                # TODO find a decent constant make weights vary giving animal area
                                weight_map[b, idx * 2, pos_y, pos_x] = 1.0 / np.sqrt(
                                    area[b, person_id]
                                )
                                weight_map[
                                    b, idx * 2 + 1, pos_y, pos_x
                                ] = 1.0 / np.sqrt(area[b, person_id])
                                area_map[b, pos_y, pos_x] = area[b, person_id]

        heatmap_weights[heatmap_weights == 2] = self.bg_weight
        return {
            "heatmap": {
                "target": torch.tensor(heatmaps, device=outputs["heatmap"].device),
                "weights": torch.tensor(
                    heatmap_weights, device=outputs["heatmap"].device
                ),
            },
            "offset": {
                "target": torch.tensor(offset_map, device=outputs["offset"].device),
                "weights": torch.tensor(weight_map, device=outputs["offset"].device),
            },
        }


def dekr_heatmap_val(sigma: float, x: float, y: float, x0: float, y0: float) -> float:
    """
    Calculates the corresponding heat value of point (x,y) given the heat distribution centered
    at (x0,y0) and spread value of sigma.

    Args:
        sigma: controls the spread or width of the heat distribution
        x: x coord of a point on the image grid
        y: y coord of a point on the image grid
        x0: x center coordinate of the heat distribution
        y0: y center coordinate of the heat distribution

    Returns:
        g: calculated heat value represents the intensity of the heat at a given position
    """
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
