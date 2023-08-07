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
from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    PREDICTORS,
    BasePredictor,
)


@PREDICTORS.register_module
class SinglePredictor(BasePredictor):
    """Predictor class for single animal pose estimation.

    Args:
        num_animals: Number of animals in the project.
        location_refinement: Enable location refinement.
        locref_stdev: Standard deviation for location refinement.
        apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.

    Returns:
        Regressed keypoints from heatmaps and locref_maps of baseline DLC model (ResNet + Deconv).
    """

    default_init = {
        "location_refinement": True,
        "locref_stdev": 7.2801,
        "apply_sigmoid": True,
    }

    def __init__(
        self,
        num_animals: int,
        location_refinement: bool,
        locref_stdev: float,
        apply_sigmoid: bool = True,
    ):
        """Initializes the SinglePredictor class.

        Args:
            num_animals: Number of animals in the project.
            location_refinement : Enable location refinement.
            locref_stdev: Standard deviation for location refinement.
            apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.

        Returns:
            None

        Notes:
            TODO: add num_animals in pytorch_cfg automatically
        """
        super().__init__()
        self.num_animals = num_animals
        assert (
            self.num_animals == 1,
            "SinglePredictor must only be used for single animal predictions",
        )
        self.location_refinement = location_refinement
        self.locref_stdev = locref_stdev
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        output: Tuple[torch.Tensor, torch.Tensor],
        scale_factors: Tuple[float, float],
    ) -> torch.Tensor:
        """Forward pass of SinglePredictor. Gets predictions from model output.

        Args:
            output: Output tensors from previous layers.
                        output = heatmaps, locref
                        heatmaps: torch.Tensor([batch_size, num_joints, height, width])
                        locref: torch.Tensor([batch_size, num_joints, height, width])
            scale_factors: Scale factors for the poses.

        Returns:
            Poses with scores.

        Example:
            >>> predictor = SinglePredictor(num_animals=1, location_refinement=True, locref_stdev=7.2801)
            >>> output = (torch.rand(32, 17, 64, 64), torch.rand(32, 17, 64, 64))
            >>> scale_factors = (0.5, 0.5)
            >>> poses = predictor.forward(output, scale_factors)
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

    def get_top_values(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top values from the heatmap.

        Args:
            heatmap: Heatmap tensor.

        Returns:
            Y and X indices of the top values.

        Example:
            >>> predictor = SinglePredictor(num_animals=1, location_refinement=True, locref_stdev=7.2801)
            >>> heatmap = torch.rand(32, 17, 64, 64)
            >>> Y, X = predictor.get_top_values(heatmap)
        """
        batchsize, ny, nx, num_joints = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)

        heatmap_top = torch.argmax(heatmap_flat, axis=1)

        Y, X = heatmap_top // nx, heatmap_top % nx
        return Y, X

    def get_pose_prediction(
        self, heatmap: torch.Tensor, locref: torch.Tensor, scale_factors
    ) -> torch.Tensor:
        """Gets the pose prediction given the heatmaps and locref.

        Args:
            heatmap: Heatmap tensor with the following format (batch_size, height, width, num_joints)
            locref: Locref tensor with the following format (batch_size, height, width, num_joints, 2)
            scale_factors: Scale factors for the poses.

        Returns:
            Pose predictions of the format: (batch_size, num_people = 1, num_joints, 3)

        Example:
            >>> predictor = SinglePredictor(num_animals=1, location_refinement=True, locref_stdev=7.2801)
            >>> heatmap = torch.rand(32, 17, 64, 64)
            >>> locref = torch.rand(32, 17, 64, 64, 2)
            >>> scale_factors = (0.5, 0.5)
            >>> poses = predictor.get_pose_prediction(heatmap, locref, scale_factors)
        """
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
    """Predictor only intended for single animal pose estimation, without locref.

    Args:
        num_animals: Number of animals in the project.
        apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.

    Returns:
        Regressed keypoints from heatmaps.
    """

    default_init = {
        "location_refinement": True,
        "locref_stdev": 7.2801,
        "apply_sigmoid": True,
    }

    def __init__(self, num_animals: int, apply_sigmoid: bool = True):
        """Initializes the HeatmapOnlyPredictor class.

        Args:
            num_animals: Number of animals in the project.
            apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.

        Returns:
            None

        Notes:
            TODO: add num_animals in pytorch_cfg automatically
        """
        super().__init__()
        self.num_animals = num_animals
        assert (
            self.num_animals == 1,
            "SinglePredictor must only be used for single animal predictions",
        )
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        output: Tuple[torch.Tensor, torch.Tensor],
        scale_factors: Tuple[float, float],
    ) -> torch.Tensor:
        """Forward pass of HeatmapOnlyPredictor. Computes predictions from the trained model output.

        Args:
            output: Output tensors from previous layers.
                    output = heatmaps
                    heatmaps: torch.Tensor([batch_size, num_joints, height, width])
                    locref: torch.Tensor([batch_size, num_joints, height, width])
            scale_factors: Scale factors for the poses.

        Returns:
            Poses with scores.

        Example:
            >>> predictor = HeatmapOnlyPredictor(num_animals=1, apply_sigmoid=True)
            >>> output = (torch.rand(32, 17, 64, 64), torch.rand(32, 17, 64, 64))
            >>> scale_factors = (0.5, 0.5)
            >>> poses = predictor.forward(output, scale_factors)
        """
        heatmaps = output[0]
        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)
        heatmaps = heatmaps.permute(0, 2, 3, 1)

        poses = self.get_pose_prediction(heatmaps, scale_factors)
        return poses

    def get_top_values(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top values from the heatmap.

        Args:
            heatmap: Heatmap tensor.

        Returns:
            Y and X indices of the top values.

        Example:
            >>> predictor = HeatmapOnlyPredictor(num_animals=1, apply_sigmoid=True)
            >>> heatmap = torch.rand(32, 17, 64, 64)
            >>> Y, X = predictor.get_top_values(heatmap)
        """
        batchsize, ny, nx, num_joints = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)

        heatmap_top = torch.argmax(heatmap_flat, axis=1)

        Y, X = heatmap_top // nx, heatmap_top % nx
        return Y, X

    def get_pose_prediction(
        self, heatmap: torch.Tensor, scale_factors: Tuple[float, float]
    ) -> torch.Tensor:
        """Get the pose prediction from heatmaps.

        Args:
            heatmap: Heatmap tensor with shape (batch_size, height, width, num_joints)
            scale_factors: Scale factors for the poses.

        Returns:
            Pose predictions following the format:  (batch_size, num_people = 1, num_joints, 3)

        Notes:
            TODO: optimize that so DZ looks right

        Example:
            >>> predictor = HeatmapOnlyPredictor(num_animals=1, apply_sigmoid=True)
            >>> heatmap = torch.rand(32, 17, 64, 64)
            >>> scale_factors = (0.5, 0.5)
            >>> poses = predictor.get_pose_prediction(heatmap, scale_factors)
        """
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
