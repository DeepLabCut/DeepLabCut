#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

from typing import Tuple

import numpy as np
import torch
from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    TARGET_GENERATORS,
    BaseGenerator,
)


@TARGET_GENERATORS.register_module
class DEKRGenerator(BaseGenerator):
    """
    Generate ground truth target for DEKR model training
    based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang
        CVPR
        2021
    Code based on:
        https://github.com/HRNet/DEKR
    """

    def __init__(self, num_joints: int, pos_dist_thresh: int, bg_weight: float = 0.1):
        """Summary:
        Constructor of the DEKRGenerator class.
        Loads the data.

        Args:
            num_joints: number of keypoints
            pos_dist_thresh: 3*std of the gaussian
            bg_weight:background weight. Defaults to 0.1.

        Returns:
            None

        Examples:
            num_joints = 6
            pos_dist_thresh = 17
            bg_weight = 0.1 (default)
        """
        super().__init__()

        self.num_joints = num_joints
        self.pos_dist_thresh = pos_dist_thresh
        self.bg_weight = bg_weight

        self.num_joints_with_center = self.num_joints + 1

    def get_heat_val(
        self, sigma: float, x: float, y: float, x0: float, y0: float
    ) -> float:
        """Summary:
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
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        return g

    def forward(
        self,
        annotations: dict,
        prediction: Tuple[torch.Tensor, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> dict:
        """Summary
        Given the annotations and predictions of your keypoints, this function returns the targets,
        a dictionary containing the heatmaps, locref_maps and locref_masks.
        Args:
            annotations: each entry should begin with the shape batch_size
            prediction: output of model, format could depend on the model, only used to compute output resolution
            image_size:size of image (only one tuple since for batch training all images should have the same size)

        Returns:
            #TODO locref is a bad name here and should be 'offset to center', but for code's simplicity it
            is easier to use the same keys as for the SingleAnimal target generators
            targets, keys:
                'heatmaps' : heatmaps
                'heatmaps_ignored': weights to apply to the heatmaps for loss computation
                'locref_maps' : offset maps
                'locref_masks' : weights to apply to the offset maps for loss computation

        Examples:
            input:
                annotations = {"keypoints":torch.randint(1,min(image_size),(batch_size, num_animals, num_joints, 2))}
                prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
                image_size = (256, 256)
            output:
                targets = {'heatmaps':scmap, 'locref_map':locref_map, 'locref_masks':locref_masks}
        """

        batch_size, _, output_h, output_w = prediction[0].shape
        output_res = output_h, output_w
        stride_y, stride_x = image_size[0] / output_h, image_size[1] / output_w

        num_joints_without_center = self.num_joints
        num_joints_with_center = num_joints_without_center + 1

        coords = annotations["keypoints"].cpu().numpy()
        num_animals = coords.shape[1]
        area = annotations["area"].cpu().numpy()

        assert (
            self.num_joints + 1 == coords.shape[2]
        ), f"the number of joints should be {coords.shape}"

        # TODO make it possible to differentiate between center sigma and other sigmas
        scale = max(1 / stride_x, 1 / stride_y)
        sgm, ct_sgm = (self.pos_dist_thresh / 2) * scale, (self.pos_dist_thresh) * scale
        radius = self.pos_dist_thresh * scale

        hms = np.zeros(
            (batch_size, num_joints_with_center, output_h, output_w), dtype=np.float32
        )
        ignored_hms = 2 * np.ones(
            (batch_size, num_joints_with_center, output_h, output_w), dtype=np.float32
        )
        offset_map = np.zeros(
            (
                batch_size,
                num_joints_without_center * 2,
                output_h,
                output_w,
            ),
            dtype=np.float32,
        )
        weight_map = np.zeros(
            (
                batch_size,
                num_joints_without_center * 2,
                output_h,
                output_w,
            ),
            dtype=np.float32,
        )
        area_map = np.zeros((batch_size, output_h, output_w), dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for b in range(batch_size):
            for person_id, p in enumerate(coords[b]):
                idx_center = len(p) - 1
                ct_x = int(p[-1, 0])
                ct_y = int(p[-1, 1])

                ct_x_sm = (ct_x - stride_x / 2) / stride_x
                ct_y_sm = (ct_y - stride_y / 2) / stride_y
                for idx, pt in enumerate(p):
                    if idx == idx_center:
                        sigma = ct_sgm
                    else:
                        sigma = sgm
                    if np.any(pt <= 0.0):
                        continue
                    x, y = pt[0], pt[1]
                    x_sm, y_sm = (x - stride_x / 2) / stride_x, (
                        y - stride_y / 2
                    ) / stride_y

                    if x_sm < 0 or y_sm < 0 or x_sm >= output_w or y_sm >= output_h:
                        continue

                    # HEATMAP COMPUTATION
                    ul = int(np.floor(x_sm - 3 * sigma - 1)), int(
                        np.floor(y_sm - 3 * sigma - 1)
                    )
                    br = int(np.ceil(x_sm + 3 * sigma + 2)), int(
                        np.ceil(y_sm + 3 * sigma + 2)
                    )

                    cc, dd = max(0, ul[0]), min(br[0], output_res[1])
                    aa, bb = max(0, ul[1]), min(br[1], output_res[0])

                    joint_rg = np.zeros((bb - aa, dd - cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(
                                sigma, sx, sy, x_sm, y_sm
                            )

                    hms_list[0][b, idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][b, idx, aa:bb, cc:dd], joint_rg
                    )
                    hms_list[1][b, idx, aa:bb, cc:dd] = 1.0

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

        hms_list[1][hms_list[1] == 2] = self.bg_weight

        targets = {
            "heatmaps": hms_list[0],
            "heatmaps_ignored": hms_list[1],
            "locref_maps": offset_map,
            "locref_masks": weight_map,
        }
        return targets
