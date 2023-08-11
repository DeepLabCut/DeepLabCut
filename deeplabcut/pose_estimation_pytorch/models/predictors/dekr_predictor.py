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

import torch
from deeplabcut.pose_estimation_pytorch.models.predictors import (
    PREDICTORS,
    BasePredictor,
)
from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
    SinglePredictor,
)


@PREDICTORS.register_module
class DEKRPredictor(BasePredictor):
    """DEKR Predictor class for multi-animal pose estimation.

    This class regresses keypoints and assembles them (if multianimal project)
    from the output of DEKR (Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression).
    Based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang
        CVPR
        2021
    Code based on:
        https://github.com/HRNet/DEKR

    Args:
        num_animals (int): Number of animals in the project.
        detection_threshold (float, optional): Threshold for detection. Defaults to 0.01.
        apply_sigmoid (bool, optional): Apply sigmoid to heatmaps. Defaults to True.
        use_heatmap (bool, optional): Use heatmap. Defaults to True.

    Attributes:
        num_animals (int): Number of animals in the project.
        detection_threshold (float): Threshold for detection.
        apply_sigmoid (bool): Apply sigmoid to heatmaps.
        use_heatmap (bool): Use heatmap.

    Example:
        # Create a DEKRPredictor instance with 2 animals.
        predictor = DEKRPredictor(num_animals=2)

        # Make a forward pass with outputs and scale factors.
        outputs = (heatmaps, offsets)  # tuple of heatmaps and offsets
        scale_factors = (0.5, 0.5)  # tuple of scale factors for the poses
        poses_with_scores = predictor.forward(outputs, scale_factors)
    """

    default_init = {"apply_sigmoid": True, "detection_threshold": 0.01}

    def __init__(
        self,
        num_animals: int,
        detection_threshold: float = 0.01,
        apply_sigmoid: bool = True,
        use_heatmap: bool = True,
        unique_bodyparts: bool = False,
    ):
        """Initializes the DEKRPredictor class.

        Args:
            num_animals: Number of animals in the project.
            detection_threshold: Threshold for detection. Defaults to 0.01.
            apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.
            use_heatmap: Use heatmap. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.num_animals = num_animals
        self.detection_threshold = detection_threshold
        self.apply_sigmoid = apply_sigmoid
        self.use_heatmap = use_heatmap
        self.unique_bodyparts = unique_bodyparts
        if self.unique_bodyparts:
            self.unique_predictor = SinglePredictor(
                num_animals=1,
                location_refinement=True,
                locref_stdev=7.8201,
                apply_sigmoid=False,
            )
        self.max_absorb_distance = 75

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        scale_factors: Tuple[float, float],
    ) -> dict:
        """Forward pass of DEKRPredictor.

        Args:
            outputs: Tuple of heatmaps and offsets.
            scale_factors: Scale factors for the poses.

        Returns:
            A dictionary containing a "poses" key with the output tensor as value, and
            optionally a "unique_bodyparts" with the unique bodyparts tensor as value.

        Example:
            # Assuming you have 'outputs' (heatmaps and offsets) and 'scale_factors' for poses
            poses_with_scores = predictor.forward(outputs, scale_factors)
        """
        if self.unique_bodyparts:
            heatmaps, offsets, unique_heatmaps, unique_locref = outputs
        else:
            heatmaps, offsets = outputs
        if self.apply_sigmoid and not self.unique_bodyparts:
            heatmaps = torch.nn.Sigmoid()(heatmaps)
        elif self.apply_sigmoid:
            heatmaps = torch.nn.Sigmoid()(heatmaps)
            unique_heatmaps = torch.nn.Sigmoid()(unique_heatmaps)

        posemap = self.offset_to_pose(offsets)

        batch_size, num_joints_with_center, h, w = heatmaps.shape
        num_joints = num_joints_with_center - 1

        center_heatmaps = heatmaps[:, -1]
        pose_ind, scores = self.get_top_values(center_heatmaps)

        posemap = posemap.permute(0, 2, 3, 1).view(batch_size, h * w, -1, 2)
        poses = torch.zeros(batch_size, pose_ind.shape[1], num_joints, 2).to(
            scores.device
        )
        for i in range(batch_size):
            pose = posemap[i, pose_ind[i]]
            poses[i] = pose

        poses = self._update_pose_with_heatmaps(poses, heatmaps[:, :-1])

        ctr_score = scores[:, :, None].expand(batch_size, -1, num_joints)[:, :, :, None]

        poses[:, :, :, 0] = (
            poses[:, :, :, 0] * scale_factors[1] + 0.5 * scale_factors[1]
        )
        poses[:, :, :, 1] = (
            poses[:, :, :, 1] * scale_factors[0] + 0.5 * scale_factors[0]
        )

        poses_w_scores = torch.cat([poses, ctr_score], dim=3)
        # self.pose_nms(heatmaps, poses_w_scores)

        if self.unique_bodyparts:
            # Super trick to compute scale factor without knowing original image size
            scale_factors_unique = (
                scale_factors[0] * h / unique_heatmaps.shape[2],
                scale_factors[0] * w / unique_heatmaps.shape[3],
            )
            unique_poses = self.unique_predictor(
                [unique_heatmaps, unique_locref], scale_factors_unique
            )

            return {
                "poses": poses_w_scores,
                "unique_bodyparts": unique_poses,
            }

        return {
            "poses": poses_w_scores,
        }

    def get_locations(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Get locations for offsets.

        Args:
            height: Height of the offsets.
            width: Width of the offsets.
            device: Device to use.

        Returns:
            Offset locations.

        Example:
            # Assuming you have 'height', 'width', and 'device'
            locations = predictor.get_locations(height, width, device)
        """
        shifts_x = torch.arange(0, width, step=1, dtype=torch.float32).to(device)
        shifts_y = torch.arange(0, height, step=1, dtype=torch.float32).to(device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)

        return locations

    def get_reg_poses(self, offsets: torch.Tensor, num_joints: int) -> torch.Tensor:
        """Get the regression poses from offsets.

        Args:
            offsets: Offsets tensor.
            num_joint: Number of joints.

        Returns:
            Regression poses.

        Example:
            # Assuming you have 'offsets' tensor and 'num_joints'
            regression_poses = predictor.get_reg_poses(offsets, num_joints)
        """
        batch_size, _, h, w = offsets.shape
        offsets = offsets.permute(0, 2, 3, 1).reshape(batch_size, h * w, num_joints, 2)
        locations = self.get_locations(h, w, offsets.device)
        locations = locations[None, :, None, :].expand(batch_size, -1, num_joints, -1)
        poses = locations - offsets

        return poses

    def offset_to_pose(self, offsets: torch.Tensor) -> torch.Tensor:
        """Convert offsets to poses.

        Args:
            offsets: Offsets tensor.

        Returns:
            Poses from offsets.

        Example:
            # Assuming you have 'offsets' tensor
            poses = predictor.offset_to_pose(offsets)
        """
        batch_size, num_offset, h, w = offsets.shape
        num_joints = int(num_offset / 2)
        reg_poses = self.get_reg_poses(offsets, num_joints)

        reg_poses = (
            reg_poses.contiguous()
            .view(batch_size, h * w, 2 * num_joints)
            .permute(0, 2, 1)
        )
        reg_poses = reg_poses.contiguous().view(batch_size, -1, h, w).contiguous()

        return reg_poses

    def max_pool(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Apply max pooling to the heatmap.

        Args:
            heatmap: Heatmap tensor.

        Returns:
            Max pooled heatmap.

        Example:
            # Assuming you have 'heatmap' tensor
            max_pooled_heatmap = predictor.max_pool(heatmap)
        """
        pool1 = torch.nn.MaxPool2d(3, 1, 1)
        pool2 = torch.nn.MaxPool2d(5, 1, 2)
        pool3 = torch.nn.MaxPool2d(7, 1, 3)
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        maxm = pool2(
            heatmap
        )  # Here I think pool 2 is a good match for default 17 pos_dist_tresh

        return maxm

    def get_top_values(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top values from the heatmap.

        Args:
            heatmap: Heatmap tensor.

        Returns:
            Position indices and scores.

        Example:
            # Assuming you have 'heatmap' tensor
            positions, scores = predictor.get_top_values(heatmap)
        """
        maximum = self.max_pool(heatmap)
        maximum = torch.eq(maximum, heatmap)
        heatmap *= maximum

        batchsize, ny, nx = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny)

        scores, pos_ind = torch.topk(heatmap_flat, self.num_animals, dim=1)

        return pos_ind, scores

    ########## WIP to take heatmap into account for scoring ##########
    def _update_pose_with_heatmaps(
        self, _poses: torch.Tensor, kpt_heatmaps: torch.Tensor
    ):
        """If a heatmap center is close enough from the regressed point, the final prediction is the center of this heatmap

        Args:
            poses: poses tensor, shape (batch_size, num_animals, num_keypoints, 2)
            kpt_heatmaps: heatmaps (does not contain the center heatmap), shape (batch_size, num_keypoints, h, w)
        """
        poses = _poses.clone()
        maxm = self.max_pool(kpt_heatmaps)
        maxm = torch.eq(maxm, kpt_heatmaps).float()
        kpt_heatmaps *= maxm
        batch_size, num_keypoints, h, w = kpt_heatmaps.shape
        kpt_heatmaps = kpt_heatmaps.view(batch_size, num_keypoints, -1)
        val_k, ind = kpt_heatmaps.topk(self.num_animals, dim=2)

        x = ind % w
        y = (ind / w).long()
        heats_ind = torch.stack((x, y), dim=3)

        for b in range(batch_size):
            for i in range(num_keypoints):
                heat_ind = heats_ind[b, i].float()
                pose_ind = poses[b, :, i]
                pose_heat_diff = pose_ind[:, None, :] - heat_ind
                pose_heat_diff.pow_(2)
                pose_heat_diff = pose_heat_diff.sum(2)
                pose_heat_diff.sqrt_()
                keep_ind = torch.argmin(pose_heat_diff, dim=1)

                for p in range(keep_ind.shape[0]):
                    if pose_heat_diff[p, keep_ind[p]] < self.max_absorb_distance:
                        poses[b, p, i] = heat_ind[keep_ind[p]]

        return poses

    def get_heat_value(
        self, pose_coords: torch.Tensor, heatmaps: torch.Tensor
    ) -> torch.Tensor:
        """Get heat values for pose coordinates and heatmaps.

        Args:
            pose_coords: Pose coordinates tensor (batch_size, num_people, num_joints, 2)
            heatmaps: Heatmaps tensor (batch_size, 1+num_joints, h, w).

        Returns:
            Heat values.

        Example:
            # Assuming you have 'pose_coords' and 'heatmaps' tensors
            heat_values = predictor.get_heat_value(pose_coords, heatmaps)
        """
        h, w = heatmaps.shape[2:]
        heatmaps_nocenter = heatmaps[:, :-1].flatten(
            2, 3
        )  # (batch_size, num_joints, h*w)

        # Predicted poses based on the offset can be outsied of the image
        y = torch.clamp(torch.floor(pose_coords[:, :, :, 1]), 0, h - 1).long()
        x = torch.clamp(torch.floor(pose_coords[:, :, :, 0]), 0, w - 1).long()

        heatvals = torch.gather(heatmaps_nocenter, 2, y * w + x)

        return heatvals

    def pose_nms(self, heatmaps: torch.Tensor, poses: torch.Tensor):
        """Non-Maximum Suppression (NMS) for regressed poses.

        Args:
            heatmaps: Heatmaps tensor.
            poses: Pose proposals.

        Returns:
            None

        Example:
            # Assuming you have 'heatmaps' and 'poses' tensors
            predictor.pose_nms(heatmaps, poses)
        """
        pose_scores = poses[:, :, :, 2]
        pose_coords = poses[:, :, :, :2]

        if pose_coords.shape[1] == 0:
            return [], []

        batch_size, num_people, num_joints, _ = pose_coords.shape
        heatvals = self.get_heat_value(pose_coords, heatmaps)
        heat_score = (torch.sum(heatvals, dim=1) / num_joints)[:, 0]

        # return heat_score
        # pose_score = pose_score*heatvals
        # poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

        # keep_pose_inds = nms_core(cfg, pose_coord, heat_score)
        # poses = poses[keep_pose_inds]
        # heat_score = heat_score[keep_pose_inds]

        # if len(keep_pose_inds) > cfg.DATASET.MAX_NUM_PEOPLE:
        #     heat_score, topk_inds = torch.topk(heat_score,
        #                                         cfg.DATASET.MAX_NUM_PEOPLE)
        #     poses = poses[topk_inds]

        # poses = [poses.numpy()]
        # scores = [i[:, 2].mean() for i in poses[0]]

        # return poses, scores
