import numpy as np
import torch
from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
from deeplabcut.pose_estimation_tensorflow.core.predict import multi_pose_predict
from torch import nn
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator


class PoseModel(nn.Module):

    def __init__(self,
                 cfg: dict,
                 backbone: torch.nn.Module,
                 head_heatmaps: torch.nn.Module,
                 head_locref: torch.nn.Module,
                 target_generator: BaseGenerator,
                 neck: torch.nn.Module = None,
                 stride: int = 8,
                ):

        super().__init__()
        self.backbone = backbone
        self.backbone.activate_batch_norm(cfg['batch_size'] >= 8) # We don't want batch norm to update for small batch sizes

        self.head_heatmaps = head_heatmaps
        self.head_locref = head_locref
        self.neck = neck
        self.stride = stride
        self.cfg = cfg
        self.target_generator = target_generator
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        TODO
        Parameters
        ----------
        x

        Returns
        -------

        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)
        heat_maps = self.head_heatmaps(features)
        loc_ref = self.head_locref(features)

        return heat_maps, loc_ref

    def get_target(self,
                   annotations,
                   prediction,
                   image_size):
        
        return self.target_generator(annotations, prediction, image_size)
