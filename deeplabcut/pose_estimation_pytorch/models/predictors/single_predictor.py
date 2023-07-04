import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    PREDICTORS,
    BasePredictor,
)


@PREDICTORS.register_module
class SinglePredictor(BasePredictor):
    """
    Predictor only intended for single animal pose estimation

    Regresses keypoints from heatmaps and locref_maps of baseline DLC model (ResNet + Deconv)
    """

    default_init = {
        "location_refinement": True,
        "locref_stdev": 7.2801,
        "apply_sigmoid": True,
    }

    def __init__(
        self, num_animals, location_refinement, locref_stdev, apply_sigmoid: bool = True
    ):
        super().__init__()
        # TODO add num_animals in pytorch_cfg automatically
        self.num_animals = num_animals
        assert (
            self.num_animals == 1,
            "SinglePredictor must only be used for single animal predictions",
        )
        self.location_refinement = location_refinement
        self.locref_stdev = locref_stdev
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, scale_factors):
        """
        get predictions from model output
        output = heatmaps, locref
        heatmaps: torch.Tensor([batch_size, num_joints, height, width])
        locref: torch.Tensor([batch_size, num_joints, height, width])
        """
        heatmaps, locrefs = output
        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)
        heatmaps = heatmaps.permute(0, 2, 3, 1)
        batch_size, height, width, num_joints = heatmaps.shape

        locrefs = locrefs.permute(0, 2, 3, 1).reshape(
            batch_size, height, width, num_joints, 2
        )

        poses = self.get_pose_prediction(
            heatmaps, locrefs * self.locref_stdev, scale_factors
        )
        return poses

    def get_top_values(self, heatmap) -> torch.Tensor:
        batchsize, ny, nx, num_joints = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)

        heatmap_top = torch.argmax(heatmap_flat, axis=1)

        Y, X = heatmap_top // nx, heatmap_top % nx
        return Y, X

    def get_pose_prediction(self, heatmap, locref, scale_factors):
        """
        heatmap shape : (batch_size, height, width, num_joints)
        locref shape : (batch_size, height, width, num_joints, 2)

        RETURN
        ----------
        pose : (batch_size, num_people = 1, num_joints, 3)"""
        Y, X = self.get_top_values(heatmap)
        batch_size, num_joints = X.shape

        DZ = torch.zeros((batch_size, 1, num_joints, 3)).to(X.device)
        for b in range(batch_size):
            for j in range(num_joints):
                DZ[b, 0, j, :2] = locref[b, Y[b, j], X[b, j], j, :]
                DZ[b, 0, j, 2] = heatmap[b, Y[b, j], X[b, j], j]

        X, Y = torch.unsqueeze(X, 1), torch.unsqueeze(Y, 1)

        X = X * scale_factors[1] + 0.5 * scale_factors[1] + DZ[:, :, :, 0]
        Y = Y * scale_factors[0] + 0.5 * scale_factors[0] + DZ[:, :, :, 1]
        # P = DZ[:, :, 2]

        pose = torch.empty((batch_size, 1, num_joints, 3))
        pose[:, :, :, 0] = X
        pose[:, :, :, 1] = Y
        pose[:, :, :, 2] = DZ[:, :, :, 2]

        return pose


@PREDICTORS.register_module
class HeatmapOnlyPredictor(BasePredictor):
    """Predictor only intended for single animal pose estimation, without locref"""

    default_init = {
        "location_refinement": True,
        "locref_stdev": 7.2801,
        "apply_sigmoid": True,
    }

    def __init__(self, num_animals, apply_sigmoid: bool = True):
        super().__init__()
        # TODO add num_animals in pytorch_cfg automatically
        self.num_animals = num_animals
        assert (
            self.num_animals == 1,
            "SinglePredictor must only be used for single animal predictions",
        )
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, scale_factors):
        """
        get predictions from model output
        output = heatmaps
        heatmaps: torch.Tensor([batch_size, num_joints, height, width])
        locref: torch.Tensor([batch_size, num_joints, height, width])
        """
        heatmaps = output[0]
        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)
        heatmaps = heatmaps.permute(0, 2, 3, 1)

        poses = self.get_pose_prediction(heatmaps, scale_factors)
        return poses

    def get_top_values(self, heatmap) -> torch.Tensor:
        batchsize, ny, nx, num_joints = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)

        heatmap_top = torch.argmax(heatmap_flat, axis=1)

        Y, X = heatmap_top // nx, heatmap_top % nx
        return Y, X

    def get_pose_prediction(self, heatmap, scale_factors):
        """
        TODO: optimize that so DZ looks right
        heatmap shape : (batch_size, height, width, num_joints)

        RETURN
        ----------
        pose : (batch_size, num_people = 1, num_joints, 3)"""
        Y, X = self.get_top_values(heatmap)
        batch_size, num_joints = X.shape

        DZ = torch.zeros((batch_size, 1, num_joints, 3)).to(X.device)
        for b in range(batch_size):
            for j in range(num_joints):
                DZ[b, 0, j, 2] = heatmap[b, Y[b, j], X[b, j], j]

        X, Y = torch.unsqueeze(X, 1), torch.unsqueeze(Y, 1)

        X = X * scale_factors[1] + 0.5 * scale_factors[1] + DZ[:, :, :, 0]
        Y = Y * scale_factors[0] + 0.5 * scale_factors[0] + DZ[:, :, :, 1]
        # P = DZ[:, :, 2]

        pose = torch.empty((batch_size, 1, num_joints, 3))
        pose[:, :, :, 0] = X
        pose[:, :, :, 1] = Y
        pose[:, :, :, 2] = DZ[:, :, :, 2]

        return pose
