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
"""Predictor to generate identity maps from head outputs"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from deeplabcut.pose_estimation_pytorch.models.predictors.base import (
    BasePredictor,
    PREDICTORS,
)


@PREDICTORS.register_module
class IdentityPredictor(BasePredictor):
    """Predictor to generate identity maps from head outputs

    Attributes:
        apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.
    """

    def __init__(self, apply_sigmoid: bool = True):
        """
        Args:
            apply_sigmoid: Apply sigmoid to heatmaps. Defaults to True.
        """
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Swaps the dimensions so the heatmap are (batch_size, h, w, num_individuals),
        optionally applies a sigmoid to the heatmaps, and rescales it to be the size
        of the original image (so that the identity scores of keypoints can be computed)

        Args:
            stride: the stride of the model
            outputs: output of the model identity head, of shape (b, num_idv, w', h')

        Returns:
            A dictionary containing a "heatmap" key with the identity heatmap tensor as
            value.
        """
        heatmaps = outputs["heatmap"]
        h_out, w_out = heatmaps.shape[2:]
        h_in, w_in = int(h_out * stride), int(w_out * stride)
        heatmaps = F.resize(
            heatmaps,
            size=[h_in, w_in],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)

        # permute to have shape (batch_size, h, w, num_individuals)
        heatmaps = heatmaps.permute((0, 2, 3, 1))
        return {"heatmap": heatmaps}
