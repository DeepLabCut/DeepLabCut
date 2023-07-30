
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

import torch.nn as nn
from .base_transformer import TokenPose_L_base
from ..builder import NECKS

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

@NECKS.register_module()
class TokenPose_L(nn.Module):

    def __init__(self,
                 IMAGE_SIZE, PATCH_SIZE, NUM_JOINTS,
                 DIM, BASE_CHANNEL, TRANSFORMER_DEPTH,
                 TRANSFORMER_HEADS, TRANSFORMER_MLP_RATIO,
                 HIDDEN_HEATMAP_DIM, HEATMAP_SIZE,
                 POS_EMBEDDING_TYPE
                 ):
        super(TokenPose_L, self).__init__()

        self.transformer = TokenPose_L_base(
                            feature_size = [IMAGE_SIZE[1]//4, IMAGE_SIZE[0]//4],
                            patch_size = [PATCH_SIZE[1], PATCH_SIZE[0]],
                            num_keypoints = NUM_JOINTS,
                            dim = DIM,
                            channels = BASE_CHANNEL,
                            depth = TRANSFORMER_DEPTH, heads = TRANSFORMER_HEADS,
                            mlp_dim = DIM*TRANSFORMER_MLP_RATIO,
                            hidden_heatmap_dim = HIDDEN_HEATMAP_DIM,
                            heatmap_dim = HEATMAP_SIZE[1]*HEATMAP_SIZE[0],
                            heatmap_size = [HEATMAP_SIZE[1], HEATMAP_SIZE[0]],
                            pos_embedding_type = POS_EMBEDDING_TYPE
        )

    def forward(self, x):                    
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained='', cfg=None):
        pass
        #self.pre_feature.init_weights(pretrained)


@NECKS.register_module()
class TokenPose_H(nn.Module):

    def __init__(self,
                 IMAGE_SIZE, PATCH_SIZE, NUM_JOINTS,
                 DIM, BASE_CHANNEL, TRANSFORMER_DEPTH,
                 TRANSFORMER_HEADS, TRANSFORMER_MLP_RATIO,
                 HIDDEN_HEATMAP_DIM, HEATMAP_SIZE,
                 POS_EMBEDDING_TYPE
                 ):
        super(TokenPose_H, self).__init__()

        self.transformer = TokenPose_L_base(
                            feature_size = [IMAGE_SIZE[1]//2, IMAGE_SIZE[0]//2],
                            patch_size = [PATCH_SIZE[1], PATCH_SIZE[0]],
                            num_keypoints = NUM_JOINTS,
                            dim = DIM,
                            channels = BASE_CHANNEL,
                            depth = TRANSFORMER_DEPTH, heads = TRANSFORMER_HEADS,
                            mlp_dim = DIM*TRANSFORMER_MLP_RATIO,
                            hidden_heatmap_dim = HIDDEN_HEATMAP_DIM,
                            heatmap_dim = HEATMAP_SIZE[1]*HEATMAP_SIZE[0],
                            heatmap_size = [HEATMAP_SIZE[1], HEATMAP_SIZE[0]],
                            pos_embedding_type = POS_EMBEDDING_TYPE
        )

    def forward(self, x):                    
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained='', cfg=None):
        pass
        #self.pre_feature.init_weights(pretrained)


        
def get_pose_net(cfg, is_train, **kwargs):
    model = TokenPose_L(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED,cfg)
    return model

