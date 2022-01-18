import torch
import torch.nn as nn
from functools import partial
import os
import pandas as pd
import pickle
import numpy as np

from mmappickle import mmapdict
from itertools import combinations


import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch._six import container_abcs
import collections.abc as container_abcs

from deeplabcut.pose_tracking_pytorch.tracking_utils import (
    load_features_from_coord,
    convert_coord_from_img_space_to_feature_space,
    query_feature_by_coord_in_img_space,
)

from deeplabcut.pose_tracking_pytorch.model import build_dlc_transformer
from .config import cfg
from deeplabcut.pose_tracking_pytorch.model.backbones import dlc_base_kpt_TransReID



inference_factory = {"dlc_transreid": dlc_base_kpt_TransReID}


class DLCTrans:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

        ckpt_dict = torch.load(self.checkpoint)

        self.model = build_dlc_transformer(
            cfg, ckpt_dict["num_kpts"], inference_factory
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        print("loading params")
        self._load_params(ckpt_dict["state_dict"])

        self.model.double()
        self.model.eval()

    def _load_params(self, params):

        self.model.load_state_dict(params)

    def _get_vec(self, inp_a, inp_b, zfill_width, feature_dict):
        coord_a_img, frame_a = inp_a
        coord_b_img, frame_b = inp_b

        frame_a = "frame" + str(frame_a).zfill(zfill_width)
        frame_b = "frame" + str(frame_b).zfill(zfill_width)

        vec_a = query_feature_by_coord_in_img_space(feature_dict, frame_a, coord_a_img)

        vec_b = query_feature_by_coord_in_img_space(feature_dict, frame_b, coord_b_img)

        return vec_a, vec_b

    def __call__(self, inp_a, inp_b, zfill_width, feature_dict, return_features=False):
        # tracklets
        device = "cuda"

        _tuple = self._get_vec(inp_a, inp_b, zfill_width, feature_dict)
        if _tuple is None:
            return None
        vec_a, vec_b = _tuple

        vec_a = np.expand_dims(vec_a, axis=0)
        vec_b = np.expand_dims(vec_b, axis=0)

        vec_a = torch.from_numpy(vec_a).double()
        vec_b = torch.from_numpy(vec_b).double()

        with torch.no_grad():
            vec_a.to(device)
            vec_b.to(device)

            vec_a = self.model(vec_a)
            vec_b = self.model(vec_b)

            dist = self.cos(vec_a, vec_b)
            if return_features:
                return dist, vec_a, vec_b
            else:
                return dist
