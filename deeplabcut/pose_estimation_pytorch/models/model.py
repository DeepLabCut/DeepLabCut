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

from typing import List, Tuple

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator


class PoseModel(nn.Module):
    """
    Complete model architecture
    """

    def __init__(
        self,
        cfg: dict,
        backbone: torch.nn.Module,
        heads: List[nn.Module],
        target_generator: BaseGenerator,
        neck: torch.nn.Module = None,
        stride: int = 8,
    ) -> None:
        """Summary
        Constructor of the PoseModel.
        Loads the data.

        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: list of head modules, one per keypoint.
            target_generator: target generator for model training
            neck: neck network architecture (default is None). Defaults to None.
            stride: stride used in the model. Defaults to 8.

        Return:
            None
        """
        super().__init__()
        self.backbone = backbone
        self.backbone.activate_batch_norm(
            cfg["batch_size"] >= 8
        )  # We don't want batch norm to update for small batch sizes

        self.heads = nn.ModuleList(heads)
        self.neck = neck
        self.stride = stride
        self.cfg = cfg
        self.target_generator = target_generator
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Summary:
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            List of output, one output per head.
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)
        outputs = []
        for head in self.heads:
            outputs.append(head(features))

        return outputs

    def get_target(
        self,
        annotations: dict,
        prediction: Tuple[torch.Tensor, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> dict:
        """Summary:
        Get targets for model training.

        Args:
            annotations: dictionary of annotations
            prediction: output of the model
                        (used here to compute the scaling factor of the model)
            image_size: image_size, used here to compute the scaling factor of the model

        Returns:
            targets: dict of the targets needed for model training
        """

        return self.target_generator(annotations, prediction, image_size)
