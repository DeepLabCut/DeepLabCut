import numpy as np
import torch
from torch import nn

from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
from deeplabcut.pose_estimation_tensorflow.core.predict import multi_pose_predict


class PoseModel(nn.Module):

    def __init__(self,
                 cfg: dict,
                 backbone: torch.nn.Module,
                 head_heatmaps: torch.nn.Module,
                 head_locref: torch.nn.Module,
                 neck: torch.nn.Module = None,
                 stride: int = 8,
                 heatmap_type: str = 'gaussian'):

        super().__init__()
        self.backbone = backbone
        self.head_heatmaps = head_heatmaps
        self.head_locref = head_locref
        self.neck = neck
        self.stride = stride
        self.cfg = cfg
        self.heatmap_type = heatmap_type
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
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)
        heat_maps = self.head_heatmaps(features)
        loc_ref = self.head_locref(features)

        return heat_maps, loc_ref

    @torch.no_grad()
    def load_model(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint), strict=True)

    def get_target(self,
                   keypoints_batch,
                   heatmap_size):

        heatmaps_target = []
        locref_target = []
        weights = []
        locref_masks = []
        for keypoints in keypoints_batch:
            # TODO: make faster
            heatmap, weight, locref_map, locref_mask = generate_heatmaps(self.cfg,
                                                                         keypoints,
                                                                         heatmap_size=heatmap_size)
            locref_target.append(locref_map)
            heatmaps_target.append(heatmap)
            locref_masks.append(locref_mask)

        heatmaps = torch.stack(heatmaps_target).permute(0, 3, 1, 2)
        locref_maps = torch.stack(locref_target).permute(0, 3, 1, 2)
        locref_masks = torch.stack(locref_masks).permute(0, 3, 1, 2)

        if weight is not None:
            weights = torch.stack(weights)
        else:
            weights = None

        target = {
            'heatmaps': heatmaps,
            'locref_maps': locref_maps,
            'locref_masks': locref_masks,
            'weights': weights
        }
        return target


    @torch.no_grad()
    def predict(self, x):
        """

        Parameters
        ----------
        x: input image tensor

        Returns
        -------
        pose: predicted keypoints coordinates
        """

        self.eval()
        poses = []
        if x.dim() == 3:
            x = x[None, :]
        heatmaps, locref = self.forward(x)
        heatmaps = self.sigmoid(heatmaps)
        heatmaps = heatmaps.permute(0, 2, 3, 1).detach().cpu().numpy()
        locref = locref.permute(0, 2, 3, 1).detach().cpu().numpy()
        for i in range(x.shape[0]):
            shape = locref[i].shape
            locref_i = np.reshape(locref, (shape[0], shape[1], -1, 2))
            if self.cfg['location_refinement']:
                locref_i = locref_i * self.cfg['locref_stdev']
            pose = multi_pose_predict(heatmaps[i], locref_i, self.stride, 1)
            poses.append(pose)
        return np.stack(poses, axis=0)
