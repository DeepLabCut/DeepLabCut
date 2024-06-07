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

from typing import Tuple

import torch

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    BasePredictor,
    PREDICTORS,
)


@PREDICTORS.register_module
class HeatmapPredictor(BasePredictor):
    """Predictor class for pose estimation from heatmaps (and optionally locrefs).

    Args:
        location_refinement: Enable location refinement.
        locref_std: Standard deviation for location refinement.
        apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.

    Returns:
        Regressed keypoints from heatmaps and locref_maps of baseline DLC model (ResNet + Deconv).
    """

    def __init__(
        self,
        apply_sigmoid: bool = True,
        clip_scores: bool = False,
        location_refinement: bool = True,
        locref_std: float = 7.2801,
    ):
        """
        Args:
            apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.
            clip_scores: If a sigmoid is not applied, this can be used to clip scores
                for predicted keypoints to values in [0, 1].
            location_refinement : Enable location refinement.
            locref_std: Standard deviation for location refinement.
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.clip_scores = clip_scores
        self.sigmoid = torch.nn.Sigmoid()
        self.location_refinement = location_refinement
        self.locref_std = locref_std

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of SinglePredictor. Gets predictions from model output.

        Args:
            stride: the stride of the model
            outputs: output of the model heads (heatmap, locref)

        Returns:
            A dictionary containing a "poses" key with the output tensor as value.

        Example:
            >>> predictor = HeatmapPredictor(location_refinement=True, locref_std=7.2801)
            >>> inputs = torch.rand((1, 3, 256, 256))
            >>> output = {"heatmap": torch.rand(32, 17, 64, 64), "locref": torch.rand(32, 17, 64, 64)}
            >>> poses = predictor.forward(inputs, output)
        """
        heatmaps = outputs["heatmap"]
        scale_factors = stride, stride

        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)

        heatmaps = heatmaps.permute(0, 2, 3, 1)
        batch_size, height, width, num_joints = heatmaps.shape

        locrefs = None
        if self.location_refinement:
            locrefs = outputs["locref"]
            locrefs = locrefs.permute(0, 2, 3, 1).reshape(
                batch_size, height, width, num_joints, 2
            )
            locrefs = locrefs * self.locref_std

        poses = self.get_pose_prediction(heatmaps, locrefs, scale_factors)

        if self.clip_scores:
            poses[..., 2] = torch.clip(poses[..., 2], min=0, max=1)

        return {"poses": poses}

    def get_top_values(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top values from the heatmap.

        Args:
            heatmap: Heatmap tensor.

        Returns:
            Y and X indices of the top values.

        Example:
            >>> predictor = HeatmapPredictor(location_refinement=True, locref_std=7.2801)
            >>> heatmap = torch.rand(32, 17, 64, 64)
            >>> Y, X = predictor.get_top_values(heatmap)
        """
        batchsize, ny, nx, num_joints = heatmap.shape
        heatmap_flat = heatmap.reshape(batchsize, nx * ny, num_joints)
        heatmap_top = torch.argmax(heatmap_flat, dim=1)
        y, x = heatmap_top // nx, heatmap_top % nx
        return y, x

    def get_pose_prediction(
        self, heatmap: torch.Tensor, locref: torch.Tensor | None, scale_factors
    ) -> torch.Tensor:
        """Gets the pose prediction given the heatmaps and locref.

        Args:
            heatmap: Heatmap tensor with the following format (batch_size, height, width, num_joints)
            locref: Locref tensor with the following format (batch_size, height, width, num_joints, 2)
            scale_factors: Scale factors for the poses.

        Returns:
            Pose predictions of the format: (batch_size, num_people = 1, num_joints, 3)

        Example:
            >>> predictor = HeatmapPredictor(location_refinement=True, locref_std=7.2801)
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
                DZ[b, 0, j, 2] = heatmap[b, Y[b, j], X[b, j], j]
                if locref is not None:
                    DZ[b, 0, j, :2] = locref[b, Y[b, j], X[b, j], j, :]

        X, Y = torch.unsqueeze(X, 1), torch.unsqueeze(Y, 1)

        X = X * scale_factors[1] + 0.5 * scale_factors[1] + DZ[:, :, :, 0]
        Y = Y * scale_factors[0] + 0.5 * scale_factors[0] + DZ[:, :, :, 1]

        pose = torch.empty((batch_size, 1, num_joints, 3))
        pose[:, :, :, 0] = X
        pose[:, :, :, 1] = Y
        pose[:, :, :, 2] = DZ[:, :, :, 2]
        return pose
