import numpy as np
from typing import Tuple
import torch

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import BaseGenerator, TARGET_GENERATORS

@TARGET_GENERATORS.register_module
class GaussianGenerator(BaseGenerator):
    """
    Generate gaussian heatmaps and locref targets from ground truth keypoints in order
    to train baseline deeplabcut model (ResNet + Deconv)
    """

    def __init__(self, locref_stdev:float, num_joints:int, pos_dist_thresh:int):
        super().__init__()

        self.locref_scale = 1.0/locref_stdev
        self.num_joints = num_joints
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh ** 2
        self.std = 2*self.dist_thresh / 3 # We think of dist_thresh as a radius and std is a 'diameter'


    def forward(self,
                annotations:dict,
                prediction:Tuple[torch.Tensor, torch.Tensor],
                image_size:Tuple[int, int]
    ):
        """

        Parameters
        ----------
        annotations: dict, each entry should begin with the shape batch_size
        prediction: output of model, format could depend on the model, only used to compute output resolution
        image_size: size of image (only one tuple since for batch training all images should have the same size)
        
        Returns
        -------
        targets : dict of the taregts, keys:
                'heatmaps' : heatmaps
                'locref_maps' : locref maps
                'locref_masks' : weights to apply to the locref maps for loss computation

        """
        # stride = cfg['stride'] # Apparently, there is no stride in the cfg
        # stride = scale_factors  # TODO just test
        batch_size, _, height, width = prediction[0].shape
        stride_y, stride_x = image_size[0]/height, image_size[1]/width
        coords = annotations['keypoints'].cpu().numpy()
        scmap = np.zeros((
            batch_size,
            height,
            width, self.num_joints), dtype=np.float32)

        locref_map = np.zeros((
            batch_size,
            height,
            width, self.num_joints * 2), dtype=np.float32)
        locref_mask = np.zeros_like(locref_map, dtype=int)

        grid = np.mgrid[:height, :width].transpose((1, 2, 0))
        grid[:, :, 0] = grid[:, :, 0] * stride_y + stride_y / 2
        grid[:, :, 1] = grid[:, :, 1] * stride_x + stride_x / 2

        for b in range(batch_size):
            for idx_animal, kpts_animal in enumerate(coords[b]):
                for i, coord in enumerate(kpts_animal):
                    coord = np.array(coord)[::-1]
                    if np.any(coord <= 0.):
                        continue
                    dist = np.linalg.norm(grid - coord, axis=2) ** 2
                    scmap_j = np.exp(-dist / (2 * self.std ** 2))
                    scmap[b, :, :, i] += scmap_j
                    locref_mask[b, dist <= self.dist_thresh_sq, i * 2:i*2+2] = 1
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