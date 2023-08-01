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
from .tokenpose_base import TokenPose_S_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class TokenPose_S(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA

        super(TokenPose_S, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.features = TokenPose_S_base(image_size=[cfg.MODEL.IMAGE_SIZE[1],cfg.MODEL.IMAGE_SIZE[0]],patch_size=[cfg.MODEL.PATCH_SIZE[1],cfg.MODEL.PATCH_SIZE[0]],
                                 num_keypoints = cfg.MODEL.NUM_JOINTS,dim =cfg.MODEL.DIM,
                                 channels=256,
                                 depth=cfg.MODEL.TRANSFORMER_DEPTH,heads=cfg.MODEL.TRANSFORMER_HEADS,
                                 mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
                                 apply_init=cfg.MODEL.INIT,
                                 hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0]//8,
                                 heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
                                 heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
                                 pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE)
        ###################################################3

    def forward(self, x):
        x = self.features(x)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            from collections import OrderedDict
            state_tmp = OrderedDict()
            for name, param in pretrained_state_dict.items():
                num = name.split(".")[1]
                if num != "19":
                    continue
                state_tmp[name] = param

            self.load_state_dict(state_tmp, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_S(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
