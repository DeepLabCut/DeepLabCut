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
class GaussianGenerator(BaseGenerator):
    """
    Generate gaussian heatmaps and locref targets from ground truth keypoints in order
    to train baseline deeplabcut model (ResNet + Deconv)
    """

    def __init__(self, locref_stdev: float, num_joints: int, pos_dist_thresh: int):
        """Summary:
        Constructor of the GaussianGenerator class.
        Loads the data.

        Args:
            locref_stdev: scaling factor
            num_joints: number of keypoints
            pos_dist_thresh: 3*std of the gaussian

        Return:
            None

        Examples:
            input:
                locref_stdev = 7.2801, default value in pytorch config
                num_joints = 6
                po_dist_thresh = 17, default value in pytorch config
        """
        super().__init__()

        self.locref_scale = 1.0 / locref_stdev
        self.num_joints = num_joints
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh**2
        self.std = (
            2 * self.dist_thresh / 3
        )  # We think of dist_thresh as a radius and std is a 'diameter'

    def forward(
        self,
        annotations: dict,
        prediction: Tuple[torch.Tensor, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> dict:
        """Summary:
        Given the annotations and predictions of your keypoints, this function returns the targets,
        a dictionary containing the heatmaps, locref_maps and locref_masks.

        Args:
            annotations: each entry should begin with the shape batch_size
            prediction: output of model format could depend on the model, only used to compute output resolution
            image_size: size of image (only one tuple since for batch training all images should have the same size)

        Returns:
            targets: dict of the taregts, keys:
                    'heatmaps' : heatmaps
                    'locref_maps' : locref maps
                    'locref_masks' : weights to apply to the locref maps for loss computation

        Examples:
            input:
                annotations = {"keypoints":torch.randint(1,min(image_size),(batch_size, num_animals, num_joints, 2))}
                prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
                image_size = (256, 256)
            output:
                targets = {'heatmaps':scmap, 'locref_map':locref_map, 'locref_masks':locref_masks}
        """

        # stride = cfg['stride'] # Apparently, there is no stride in the cfg
        # stride = scale_factors  # TODO just test
        batch_size, _, height, width = prediction[0].shape
        stride_y, stride_x = image_size[0] / height, image_size[1] / width
        coords = annotations["keypoints"].cpu().numpy()
        scmap = np.zeros((batch_size, height, width, self.num_joints), dtype=np.float32)

        locref_map = np.zeros(
            (batch_size, height, width, self.num_joints * 2), dtype=np.float32
        )
        locref_mask = np.zeros_like(locref_map, dtype=int)

        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid[:, :, 0] = grid[:, :, 0] * stride_y + stride_y / 2
        grid[:, :, 1] = grid[:, :, 1] * stride_x + stride_x / 2

        for b in range(batch_size):
            for idx_animal, kpts_animal in enumerate(coords[b]):
                for i, coord in enumerate(kpts_animal):
                    coord = np.array(coord)[::-1]
                    if np.any(coord <= 0.0):
                        continue
                    dist = np.linalg.norm(grid - coord, axis=2) ** 2
                    scmap_j = np.exp(-dist / (2 * self.std**2))
                    scmap[b, :, :, i] += scmap_j
                    locref_mask[b, dist <= self.dist_thresh_sq, i * 2 : i * 2 + 2] = 1
                    dx = coord[1] - grid.copy()[:, :, 1]
                    dy = coord[0] - grid.copy()[:, :, 0]
                    locref_map[b, :, :, i * 2 + 0] += dx * self.locref_scale
                    locref_map[b, :, :, i * 2 + 1] += dy * self.locref_scale
        scmap = scmap.transpose(0, 3, 1, 2)
        locref_map = locref_map.transpose(0, 3, 1, 2)
        locref_mask = locref_mask.transpose(0, 3, 1, 2)
        targets = {
            "heatmaps": scmap,
            "locref_maps": locref_map,
            "locref_masks": locref_mask,
        }

        return targets


@TARGET_GENERATORS.register_module
class GaussianWithoutLocref(BaseGenerator):
    """
    Generate plateau heatmaps from ground truth keypoints in order
    to train baseline deeplabcut model (ResNet + Deconv)
    """

    def __init__(self, num_joints: int, pos_dist_thresh: int):
        """Summary:
        Constructor of the GaussianWithoutLocref class.
        Loads the data.

        Args:
            num_joints: number of keypoints
            pos_dist_thresh: 3*std of the gaussian

        Returns:
            None

        Examples:
            input:
                num_joints = 6
                po_dist_thresh = 17, default value in pytorch config
        """
        super().__init__()

        self.num_joints = num_joints
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh**2
        self.std = 2 * self.dist_thresh / 3

    def forward(
        self,
        annotations: dict,
        prediction: Tuple[torch.Tensor, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> dict:
        """Summary:
        Given the annotations and predictions of your keypoints, this function returns the targets,
        a dictionary containing the heatmaps, locref_maps and locref_masks.

        Args:
            annotations: dict of annoations which should all be tensors of first dimension batch_size
            prediction: model's output
            image_size: size of input images

        Returns:
            input:
                annotations = {"keypoints":torch.randint(1,min(image_size),(batch_size, num_animals, num_joints, 2))}
                prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
                image_size = (256, 256)
            output:
                targets = {'heatmaps':scmap, 'locref_map':locref_map, 'locref_masks':locref_masks}

        """
        batch_size, _, height, width = prediction[0].shape
        stride_y, stride_x = image_size[0] / height, image_size[1] / width
        coords = annotations["keypoints"].cpu().numpy()
        scmap = np.zeros((batch_size, height, width, self.num_joints), dtype=np.float32)

        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid[:, :, 0] = grid[:, :, 0] * stride_y + stride_y / 2
        grid[:, :, 1] = grid[:, :, 1] * stride_x + stride_x / 2

        for b in range(batch_size):
            for idx_animal, kpts_animal in enumerate(coords[b]):
                for i, coord in enumerate(kpts_animal):
                    coord = np.array(coord)[::-1]
                    if np.any(coord <= 0.0):
                        continue
                    dist = np.linalg.norm(grid - coord, axis=2) ** 2
                    scmap_j = np.exp(-dist / (2 * self.std**2))
                    scmap[b, :, :, i] += scmap_j

        scmap = scmap.transpose(0, 3, 1, 2)
        targets = {
            "heatmaps": scmap,
        }

        return targets
