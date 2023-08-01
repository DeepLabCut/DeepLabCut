# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
# import timm
import math
from .tokenpose_base import TokenPose_TB_base
from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class TokenPose_B(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA

        super(TokenPose_B, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.pre_feature = HRNET_base(cfg,**kwargs)
        self.transformer = TokenPose_TB_base(feature_size=[cfg.MODEL.IMAGE_SIZE[1]//4,cfg.MODEL.IMAGE_SIZE[0]//4],patch_size=[cfg.MODEL.PATCH_SIZE[1],cfg.MODEL.PATCH_SIZE[0]],
                                 num_keypoints = cfg.MODEL.NUM_JOINTS,dim =cfg.MODEL.DIM,
                                 channels=cfg.MODEL.BASE_CHANNEL,
                                 depth=cfg.MODEL.TRANSFORMER_DEPTH,heads=cfg.MODEL.TRANSFORMER_HEADS,
                                 mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
                                 apply_init=cfg.MODEL.INIT,
                                 hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0]//8,
                                 heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
                                 heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
                                 pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE)
        ###################################################3

    def forward(self, x):
        x = self.pre_feature(x)
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_B(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
