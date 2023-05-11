import numpy as np

from deeplabcut.pose_estimation_pytorch.models.target_generators.base import BaseGenerator, TARGET_GENERATORS

@TARGET_GENERATORS.register_module
class PlateauGenerator(BaseGenerator):

    def __init__(self, locref_stdev, num_joints, pos_dist_thresh):
        super().__init__()
        
        self.locref_scale = 1.0/locref_stdev
        self.num_joints = num_joints
        self.dist_thresh = float(pos_dist_thresh)
        self.dist_thresh_sq = self.dist_thresh ** 2

    def forward(self, annotations, prediction, image_size):
        """

        Parameters
        ----------
        annotations : dict of annoations which should all be tensors of first dimension batch_size
        prediction: model's output
        image_size : size of input images

        Returns
        -------

        """
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
                    mask = (dist <= self.dist_thresh_sq)
                    scmap[b, (dist <= self.dist_thresh_sq), i] += 1
                    locref_mask[b, dist <= self.dist_thresh_sq, i * 2:i*2+2] = 1
                    dx = coord[1] - grid.copy()[:, :, 1]
                    dy = coord[0] - grid.copy()[:, :, 0]
                    locref_map[b, mask, i * 2 + 0] += (dx * self.locref_scale)[mask]
                    locref_map[b, mask, i * 2 + 1] += (dy * self.locref_scale)[mask]

        scmap = scmap.transpose(0, 3, 1, 2)
        locref_map = locref_map.transpose(0, 3, 1, 2)
        locref_mask = locref_mask.transpose(0, 3, 1, 2)
        targets = {
            "heatmaps": scmap,
            "locref_maps": locref_map,
            "locref_masks": locref_mask,
        }

        return targets