import numpy as np
import torch
from typing import Tuple
from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
from deeplabcut.pose_estimation_tensorflow.core.predict import multi_pose_predict
from torch import nn
from typing import List
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
    ):
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
        """
        TODO
        Parameters
        ----------
        x: input images

        Returns
        -------
            outputs : list of outputs, one output per head
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
    ):
        """_summary_

        Args:
            annotations (dict): dict of annotations
            prediction (Tuple[torch.Tensor, torch.Tensor]): output of the model
                        (used here to compute the scaling factor of the model)
            image_size (Tuple[int, int]): image_size, used here to compute the scaling factor of the model

        Returns:
            targets : dict of the targets needed for model training
        """

        return self.target_generator(annotations, prediction, image_size)
