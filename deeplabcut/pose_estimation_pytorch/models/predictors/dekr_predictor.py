import torch
import torch.nn as nn

from typing import Tuple

from deeplabcut.pose_estimation_pytorch.models.predictors import (
    PREDICTORS,
    BasePredictor,
)


@PREDICTORS.register_module
class DEKRPredictor(BasePredictor):
    """
    Regresses keypoints and assembles them (if multianimal project) from DEKR output
    Based on:
        Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression
        Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, Jingdong Wang
        CVPR
        2021
    Code based on:
        https://github.com/HRNet/DEKR
    """

    default_init = {"apply_sigmoid": True, "detection_threshold": 0.01}

    def __init__(
        self,
        num_animals: int,
        detection_threshold: float = 0.01,
        apply_sigmoid: bool = True,
        use_heatmap=True,
    ):
        super().__init__()

        self.num_animals = num_animals
        self.detection_threshold = detection_threshold
        self.apply_sigmoid = apply_sigmoid
        self.use_heatmap = use_heatmap

    def forward(self, outputs, scale_factors: Tuple[float, float]):
        # TODO implement confidence scores for each keypoints
        heatmaps, offsets = outputs
        if self.apply_sigmoid:
            heatmaps = nn.Sigmoid()(heatmaps)
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

        ctr_score = scores[:, :, None].expand(batch_size, -1, num_joints)[:, :, :, None]

        poses[:, :, :, 0] = (
            poses[:, :, :, 0] * scale_factors[1] + 0.5 * scale_factors[1]
        )
        poses[:, :, :, 1] = (
            poses[:, :, :, 1] * scale_factors[0] + 0.5 * scale_factors[0]
        )

        poses_w_scores = torch.cat([poses, ctr_score], dim=3)
        # self.pose_nms(heatmaps, poses_w_scores)

        return poses_w_scores

    def get_locations(self, height: int, width: int, device: torch.device):
        shifts_x = torch.arange(0, width, step=1, dtype=torch.float32).to(device)
        shifts_y = torch.arange(0, height, step=1, dtype=torch.float32).to(device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)

        return locations

    def get_reg_poses(self, offsets: torch.Tensor, num_joints: int):
        """
        offsets : (batch_size, num_joints*2, h, w)
        """
        batch_size, _, h, w = offsets.shape
        offsets = offsets.permute(0, 2, 3, 1).reshape(batch_size, h * w, num_joints, 2)
        locations = self.get_locations(h, w, offsets.device)
        locations = locations[None, :, None, :].expand(batch_size, -1, num_joints, -1)
        poses = locations - offsets

        return poses

    def offset_to_pose(self, offsets: torch.Tensor):
        """
        offsets : (batch_size, num_joints*2, h, w)

        RETURN
        ---------
        reg_poses : (batch_size, 2*num_joints, h, w)
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

    def max_pool(self, heatmap: torch.Tensor):
        """
        heatmap: (batch_size, h, w)
        """
        pool1 = torch.nn.MaxPool2d(3, 1, 1)
        pool2 = torch.nn.MaxPool2d(5, 1, 2)
        pool3 = torch.nn.MaxPool2d(7, 1, 3)
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        maxm = pool2(
            heatmap
        )  # Here I think pool 2 is a good match for default 17 pos_dist_tresh

        return maxm

    def get_top_values(self, heatmap: torch.Tensor):
        """
        heatmap: (batch_size, h, w)
        """
        maximum = self.max_pool(heatmap)
        maximum = torch.eq(maximum, heatmap)
        heatmap *= maximum

        batchsize, ny, nx = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny)

        scores, pos_ind = torch.topk(heatmap_flat, self.num_animals, dim=1)

        return pos_ind, scores

    ########## WIP to take heatmap into account for scoring ##########
    def get_heat_value(self, pose_coords, heatmaps):
        """
        pose_coords : (batch_size, num_people, num_joints, 2)
        heatmaps : (batch_size, 1+num_joints, h, w)
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

    def pose_nms(self, heatmaps, poses):
        """
        NMS for the regressed poses results.

        Args:
            heatmaps (Tensor): Avg of the heatmaps at all scales (batch_size, 1+num_joints, h, w)
            poses (List): Gather of the pose proposals (batch_size, num_people, num_joints, 3)
        """
        pose_scores = poses[:, :, :, 2]
        pose_coords = poses[:, :, :, :2]

        if pose_coords.shape[1] == 0:
            return [], []

        batch_size, num_people, num_joints, _ = pose_coords.shape
        heatvals = self.get_heat_value(pose_coords, heatmaps)
        heat_score = (torch.sum(heatvals, dim=1) / num_joints)[:, 0]

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
