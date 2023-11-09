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
from __future__ import annotations

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    BaseGenerator,
    TARGET_GENERATORS,
)


@TARGET_GENERATORS.register_module
class GaussianGenerator(BaseGenerator):
    """
    TODO: Remove code duplication with PlateauGenerator

    Generate gaussian heatmaps and locref targets from ground truth keypoints in order
    to train baseline deeplabcut model (ResNet + Deconv)
    """

    def __init__(
        self, locref_stdev: float, num_joints: int, pos_dist_thresh: int, **kwargs
    ):
        """
        Args:
            locref_stdev: scaling factor
            num_joints: number of keypoints
            pos_dist_thresh: 3*std of the gaussian

        Examples:
            input:
                locref_stdev = 7.2801, default value in pytorch config
                num_joints = 6
                po_dist_thresh = 17, default value in pytorch config
        """
        super().__init__(**kwargs)
        self.locref_scale = 1.0 / locref_stdev
        self.num_joints = num_joints
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh ** 2
        self.std = (
            2 * self.dist_thresh / 3
        )  # We think of dist_thresh as a radius and std is a 'diameter'

    def forward(
        self, inputs: torch.Tensor, outputs: torch.Tensor, labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Summary:
        Given the annotations and predictions of your keypoints, this function returns the targets,
        a dictionary containing the heatmaps, locref_maps and locref_masks.

        Args:
            inputs: the input images given to the model, of shape (b, c, w, h)
            outputs: output of each model head
            labels: the labels for the inputs (each tensor should have shape (b, ...))

        Returns:
            The targets for the heatmap and locref heads:
                {
                    "heatmap": {
                        "target": heatmaps,
                        "weights":  heatmap_weights,
                    },
                    "locref": {
                        "target": locref_map,
                        "weights": locref_weights,
                    }
                }

        Examples:
            input:
                annotations = {"keypoints":torch.randint(1,min(image_size),(batch_size, num_animals, num_joints, 2))}
                prediction = [torch.rand((batch_size, num_joints, image_size[0], image_size[1]))]
                image_size = (256, 256)
            output:
                targets = {'heatmaps':scmap, 'locref_map':locref_map, 'locref_masks':locref_masks}
        """
        batch_size, _, input_h, input_w = inputs.shape
        height, width = outputs.shape[2:]
        stride_y, stride_x = input_h / height, input_w / width
        coords = labels[self.label_keypoint_key].cpu().numpy()
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
                    scmap_j = np.exp(-dist / (2 * self.std ** 2))
                    scmap[b, :, :, i] += scmap_j
                    locref_mask[b, dist <= self.dist_thresh_sq, i * 2 : i * 2 + 2] = 1
                    dx = coord[1] - grid.copy()[:, :, 1]
                    dy = coord[0] - grid.copy()[:, :, 0]
                    locref_map[b, :, :, i * 2 + 0] += dx * self.locref_scale
                    locref_map[b, :, :, i * 2 + 1] += dy * self.locref_scale
        scmap = scmap.transpose(0, 3, 1, 2)
        locref_map = locref_map.transpose(0, 3, 1, 2)
        locref_mask = locref_mask.transpose(0, 3, 1, 2)
        return {
            "heatmap": {
                "target": torch.tensor(scmap, device=outputs["heatmap"].device)
            },
            "locref": {
                "target": torch.tensor(locref_map, device=outputs["locref"].device),
                "weights": torch.tensor(locref_mask, device=outputs["locref"].device),
            },
        }
