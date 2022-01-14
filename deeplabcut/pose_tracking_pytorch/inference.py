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


"""
class build_dlc_transformer(nn.Module):
    def __init__(self, kpt_num, embed_dim = 128, depth = 4, num_heads = 4, mlp_ratio = 1, attn_drop_rate = 0.0, drop_rate = 0.0, sie_xishu = 1.5, drop_path_rate = 0.1, local_feature = False):
        super(build_dlc_transformer,self).__init__()
        self.in_planes = 128
        self.num_kpts = kpt_num

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.sie_xishu = sie_xishu
        self.attn_drop_rate = attn_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.local_feature = local_feature
        
        self.base = DLCTransReID(embed_dim = self.embed_dim,
                                 depth = self.depth,
                                 num_heads = self.num_heads,
                                 mlp_ratio = self.mlp_ratio,
                                 qkv_bias = True,
                                 norm_layer = partial(nn.LayerNorm, eps = 1e-6),
                                 local_feature = self.local_feature,                                 
                                 sie_xishu=self.sie_xishu,
                                 drop_path_rate=self.drop_path_rate,
                                 drop_rate= self.drop_rate,
                                 attn_drop_rate=self.attn_drop_rate,
                                 num_kpts = self.num_kpts)
        
        
        self.classifier = nn.Identity()        

        self.bottleneck = nn.Identity()        

    def forward(self, x):
        global_feat = self.base(x)

        feat = self.bottleneck(global_feat)
        
        q = self.classifier(feat)
        norm = torch.norm(q, p=2, dim=1, keepdim = True)
        q = q.div(norm)
        return q

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
"""


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
