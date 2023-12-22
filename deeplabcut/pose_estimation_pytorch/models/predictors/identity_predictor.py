"""Predictor to generate identity maps from head outputs


"""
from __future__ import annotations

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
        self, inputs: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of IdentityPredictor.

        Swaps the dimensions so the heatmap are (batch_size, h, w, num_individuals),
        optionally applies a sigmoid to the heatmaps, and rescales it to be the size
        of the original image (so that the identity scores of keypoints can be computed)

        Args:
            inputs: the input images given to the model, of shape (b, c, h, w)
            outputs: output of the model identity head, of shape (b, num_individuals, w', h')

        Returns:
            A dictionary containing a "heatmap" key with the identity heatmap tensor as
            value.
        """
        heatmaps = outputs["heatmap"]
        h_in, w_in = inputs.shape[2:]
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
