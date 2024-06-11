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

from abc import abstractmethod
from enum import Enum

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import (
    BaseGenerator,
    TARGET_GENERATORS,
)


class HeatmapGenerator(BaseGenerator):
    """Abstract class to generate target heatmap targets (with/without locref)

    Can generate target heatmaps either for pose estimation (one keypoint), or for
    individual identification.

    This class is abstract, and heatmap targets should be generated through its
    subclasses (such as HeatmapPlateauGenerator)
    """

    class Mode(Enum):
        """
        KEYPOINT generates one heatmap per type of keypoint (for pose estimation heads)
        INDIVIDUAL generates one heatmap per individual (for identification heads)
        """

        INDIVIDUAL = "INDIVIDUAL"
        KEYPOINT = "KEYPOINT"

        @classmethod
        def _missing_(cls, value):
            if isinstance(value, str):
                value = value.upper()
                for member in cls:
                    if member.value == value:
                        return member
            return None

    def __init__(
        self,
        num_heatmaps: int,
        pos_dist_thresh: int,
        heatmap_mode: str | Mode = Mode.KEYPOINT,
        generate_locref: bool = True,
        locref_std: float = 7.2801,
        **kwargs,
    ):
        """
        Args:
            num_heatmaps: the number of heatmaps to generate
            pos_dist_thresh: 3*std of the gaussian. We think of dist_thresh as a radius
                and std is a 'diameter'.
            mode: the mode to generate heatmaps for
            learned_id_target: whether to generate the heatmap for keypoints
                or for learned IDs
            generate_locref: whether to generate location refinement maps
            locref_std: the STD for the location refinement maps, if defined

        Examples:
            input:
                locref_std = 7.2801, default value in pytorch config
                num_joints = 6
                po_dist_thresh = 17, default value in pytorch config
        """
        super().__init__(**kwargs)
        self.num_heatmaps = num_heatmaps
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh**2
        self.std = 2 * self.dist_thresh / 3

        if isinstance(heatmap_mode, str):
            heatmap_mode = HeatmapGenerator.Mode(heatmap_mode)
        self.heatmap_mode = heatmap_mode

        self.generate_locref = generate_locref
        self.locref_scale = 1.0 / locref_std

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
            The targets for the heatmap and locref heads:
                {
                    "heatmap": {
                        "target": heatmaps,
                        "weights":  heatmap_weights,
                    },
                    "locref": {  # optional
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
        stride_y, stride_x = stride, stride
        batch_size, _, height, width = outputs["heatmap"].shape
        coords = labels[self.label_keypoint_key].cpu().numpy()
        if len(coords.shape) == 3:  # for single animal: add individual dimension
            coords = coords.reshape((batch_size, 1, *coords.shape[1:]))

        if self.heatmap_mode == HeatmapGenerator.Mode.KEYPOINT:
            # transpose the individuals and keypoints to iterate over bodyparts
            coords = coords.transpose((0, 2, 1, 3))
        if self.heatmap_mode == HeatmapGenerator.Mode.INDIVIDUAL:
            # re-order the individuals to always have the same order
            # TODO: Optimize
            sorted_coords = -np.ones_like(coords)
            for i, batch_individuals in enumerate(labels["individual_ids"]):
                for j, individual_id in enumerate(batch_individuals):
                    if individual_id >= 0:
                        sorted_coords[i, individual_id] = coords[i, j]
            coords = sorted_coords

        map_size = batch_size, height, width
        heatmap = np.zeros((*map_size, self.num_heatmaps), dtype=np.float32)

        # coords shape: (batch_size, n_keypoints, 1, 2)
        weights = np.ones(
            (batch_size, coords.shape[1], height, width), dtype=np.float32
        )

        locref_map, locref_mask = None, None
        if self.generate_locref:
            locref_map = np.zeros((*map_size, self.num_heatmaps * 2), dtype=np.float32)
            locref_mask = np.zeros_like(locref_map, dtype=int)

        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid[:, :, 0] = grid[:, :, 0] * stride_y + stride_y / 2
        grid[:, :, 1] = grid[:, :, 1] * stride_x + stride_x / 2

        # heatmap (batch_size, height, width, num_kpts)
        # coords (batch_size, num_kpts, num_individuals, 3)
        for b in range(batch_size):
            for heatmap_idx, group_keypoints in enumerate(coords[b]):
                for keypoint in group_keypoints:
                    # FIXME: Gradient masking weights should be parameters
                    if keypoint[-1] == 0:
                        # full gradient masking
                        weights[b, heatmap_idx] = 0.0
                    elif keypoint[-1] == -1:
                        # full gradient masking
                        weights[b, heatmap_idx] = 0.0

                    elif keypoint[-1] > 0:
                        # keypoint visible
                        self.update(
                            heatmap=heatmap[b, :, :, heatmap_idx],
                            grid=grid,
                            keypoint=keypoint[..., :2],
                            locref_map=self.get_locref(locref_map, b, heatmap_idx),
                            locref_mask=self.get_locref(locref_mask, b, heatmap_idx),
                        )

        hm_device = outputs["heatmap"].device
        heatmap = heatmap.transpose((0, 3, 1, 2))
        target = {
            "heatmap": {
                "target": torch.tensor(heatmap, device=hm_device),
                "weights": torch.tensor(weights, device=hm_device),
            }
        }

        # we don't handle masking for locref
        if self.generate_locref:
            locref_map = locref_map.transpose((0, 3, 1, 2))
            locref_mask = locref_mask.transpose((0, 3, 1, 2))
            target["locref"] = {
                "target": torch.tensor(locref_map, device=outputs["locref"].device),
                "weights": torch.tensor(locref_mask, device=outputs["locref"].device),
            }

        return target

    def get_locref(
        self,
        locref_map_or_mask: np.ndarray | None,
        batch_idx: int,
        heatmap_idx: int,
    ) -> np.ndarray | None:
        """
        Args:
            locref_map_or_mask: the locref array to return (either the map or mask), of
                shape (batch_size, height, width, num_heatmaps)
            batch_idx: the index of the batch
            heatmap_idx: the index of the heatmap for which we want the location
                refinement maps or masks

        Returns:
            the location refinement maps/masks of shape (height, width, 2)
        """
        if not self.generate_locref:
            return None

        start_idx = 2 * heatmap_idx
        end_idx = start_idx + 2
        return locref_map_or_mask[batch_idx, :, :, start_idx:end_idx]

    @abstractmethod
    def update(
        self,
        heatmap: np.ndarray,
        grid: np.mgrid,
        keypoint: np.ndarray,
        locref_map: np.ndarray | None,
        locref_mask: np.ndarray | None,
    ) -> None:
        """
        Updates the heatmap and locref targets in-place following an update rule (e.g.,
        Gaussian or Plateau).

        Args:
            heatmap: the heatmap to update of shape (height, width)
            grid: the grid for ???
            keypoint: the keypoint with which to update the maps
            locref_map: the location refinement maps of shape (height, width, 2), if
                self.generate_locref = True
            locref_mask: the location refinement masks of shape (height, width, 2), if
                self.generate_locref = True
        """
        raise NotImplementedError


@TARGET_GENERATORS.register_module
class HeatmapGaussianGenerator(HeatmapGenerator):
    """Generates gaussian heatmaps (and locref) targets from keypoints"""

    def update(
        self,
        heatmap: np.ndarray,
        grid: np.mgrid,
        keypoint: np.ndarray,
        locref_map: np.ndarray | None,
        locref_mask: np.ndarray | None,
    ) -> None:
        """Updates the heatmap (and locref if defined) with gaussian values"""
        # revert keypoints to follow image convention: from x,y to y,x
        keypoint = keypoint.copy()[::-1]

        dist = np.linalg.norm(grid - keypoint, axis=2) ** 2
        heatmap_j = np.exp(-dist / (2 * self.std**2))
        heatmap[:, :] = np.maximum(heatmap, heatmap_j)

        if locref_map is not None:
            dx = keypoint[1] - grid.copy()[:, :, 1]
            dy = keypoint[0] - grid.copy()[:, :, 0]
            locref_map[:, :, 0] = dx * self.locref_scale
            locref_map[:, :, 1] = dy * self.locref_scale

        if locref_mask is not None:
            locref_mask[dist <= self.dist_thresh_sq] = 1


@TARGET_GENERATORS.register_module
class HeatmapPlateauGenerator(HeatmapGenerator):
    """Generates plateau heatmaps (and locref) targets from keypoints"""

    def update(
        self,
        heatmap: np.ndarray,
        grid: np.mgrid,
        keypoint: np.ndarray,
        locref_map: np.ndarray | None,
        locref_mask: np.ndarray | None,
    ) -> None:
        """Updates the heatmap (and locref if defined) with plateau values"""
        # revert keypoints to follow image convention: from x,y to y,x
        keypoint = keypoint.copy()[::-1]
        dist = np.sum((grid - keypoint) ** 2, axis=2)
        mask = dist <= self.dist_thresh_sq
        heatmap[mask] = 1

        if locref_map is not None:
            dx = keypoint[1] - grid.copy()[:, :, 1]
            dy = keypoint[0] - grid.copy()[:, :, 0]
            locref_map[mask, 0] = (dx * self.locref_scale)[mask]
            locref_map[mask, 1] = (dy * self.locref_scale)[mask]

        if locref_mask is not None:
            locref_mask[mask] = 1
