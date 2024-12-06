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

from math import sqrt

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    BaseGenerator,
    TARGET_GENERATORS,
)


@TARGET_GENERATORS.register_module
class PartAffinityFieldGenerator(BaseGenerator):
    """
    Generate part affinity field targets from ground truth keypoints in order
    to train baseline multi-animal deeplabcut model (ResNet + Deconv)
    """

    def __init__(self, graph: list[list[int, int]], width: float):
        """
        Args:
            graph: list of pairs of keypoint indices forming
                the graph edges
            width: width of the vector field in pixels

        Examples:
            input:
                graph = [(0, 1), (0, 2), (1, 2)]
                width = 20.0, default value in pytorch config
        """
        super().__init__()
        self.graph = graph
        self.width = width
        self.num_limbs = len(graph)

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor], labels: dict
    ) -> dict[str, dict[str, torch.Tensor]]:
        stride_y, stride_x = stride, stride
        batch_size, _, height, width = outputs["heatmap"].shape
        coords = labels[self.label_keypoint_key].cpu().numpy()

        paf_map = np.zeros(
            (batch_size, height, width, self.num_limbs * 2), dtype=np.float32
        )
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid[:, :, 0] = grid[:, :, 0] * stride_y + stride_y / 2
        grid[:, :, 1] = grid[:, :, 1] * stride_x + stride_x / 2
        y, x = np.rollaxis(grid, 2)

        for b in range(batch_size):
            for _, kpts_animal in enumerate(coords[b]):
                visible = set(np.flatnonzero(kpts_animal[..., -1] > 0))
                kpts_animal = kpts_animal[..., :2]
                for l, (bp1, bp2) in enumerate(self.graph):
                    if not (bp1 in visible and bp2 in visible):
                        continue

                    j1_x, j1_y = kpts_animal[bp1]
                    j2_x, j2_y = kpts_animal[bp2]
                    vec_x = j2_x - j1_x
                    vec_y = j2_y - j1_y
                    dist = sqrt(vec_x ** 2 + vec_y ** 2)
                    if dist > 0:
                        vec_x_norm = vec_x / dist
                        vec_y_norm = vec_y / dist
                        vec = [
                            vec_x_norm * j1_x + vec_y_norm * j1_y,
                            vec_x_norm * j2_x + vec_y_norm * j2_y,
                        ]
                        vec_ortho = j1_y * vec_x_norm - j1_x * vec_y_norm

                        distance_along = vec_x_norm * x + vec_y_norm * y
                        distance_across = (
                            ((y * vec_x_norm - x * vec_y_norm) - vec_ortho)
                            * 1.0
                            / self.width
                        )

                        mask1 = (distance_along >= min(vec)) & (
                            distance_along <= max(vec)
                        )
                        distance_across_abs = np.abs(distance_across)
                        mask2 = distance_across_abs <= 1
                        mask = mask1 & mask2
                        temp = 1 - distance_across_abs[mask]
                        paf_map[b, mask, l * 2 + 0] = vec_x_norm * temp
                        paf_map[b, mask, l * 2 + 1] = vec_y_norm * temp

        paf_map = paf_map.transpose((0, 3, 1, 2))
        return {
            "paf": {
                "target": torch.tensor(
                    paf_map, device=outputs["paf"].device
                )
            }
        }
