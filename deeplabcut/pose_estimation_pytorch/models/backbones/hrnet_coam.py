# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)
from deeplabcut.pose_estimation_pytorch.models.modules import (
    BasicBlock,
    Bottleneck,
    HighResolutionModule,
    CoAMBlock,
    SelfAttentionModule_CoAM,
)


logger = logging.getLogger(__name__)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


@BACKBONES.register_module
class HRNet_CoAM(BaseBackbone):
    """HRNet backbone with Conditional Attention Module (CoAM).

    This version returns high-resolution feature maps of size 1/4 * original_image_size.
    """

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(HRNet_CoAM, self).__init__()

        self.cfg = cfg

        self.bn_momentum = 0.1

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # ------------------------------------------------

        att_heads = self.cfg['MODEL']['ATTENTION_HEADS']

        self.stage1_att = None
        self.stage2_att = None
        self.stage3_att = None
        self.stage4_att = None
        self.att_config = cfg.MODEL.ATT_MODULES
        self.selfatt_config = cfg.MODEL.SELFATT_MODULES

        spat_dims = [(int(cfg.MODEL.IMAGE_SIZE[0]/4),int(cfg.MODEL.IMAGE_SIZE[1]/4)),
                        (int(cfg.MODEL.IMAGE_SIZE[0]/8),int(cfg.MODEL.IMAGE_SIZE[1]/8)),
                        (int(cfg.MODEL.IMAGE_SIZE[0]/16),int(cfg.MODEL.IMAGE_SIZE[1]/16)),
                        (int(cfg.MODEL.IMAGE_SIZE[0]/32),int(cfg.MODEL.IMAGE_SIZE[1]/32))]

        assert not self.att_config[0] or not self.selfatt_config[0]
        assert not self.att_config[1] or not self.selfatt_config[1]
        assert not self.att_config[2] or not self.selfatt_config[2]
        assert not self.att_config[3] or not self.selfatt_config[3]

        if self.att_config[0]:
            self.stage1_att = CoAMBlock(spat_dims=spat_dims[:2], channel_list=self.stage2_cfg['NUM_CHANNELS'],
                                                cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'])
        if self.att_config[1]:
            self.stage2_att = CoAMBlock(spat_dims=spat_dims[:3], channel_list=self.stage3_cfg['NUM_CHANNELS'],
                                                cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'])
        if self.att_config[2]:
            self.stage3_att = CoAMBlock(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS'],
                                                cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'])
        if self.att_config[3]:
            self.stage4_att = CoAMBlock(spat_dims=[spat_dims[0]], channel_list=[self.stage4_cfg['NUM_CHANNELS'][0]],
                                                cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'])

        if self.selfatt_config[0]:
            self.stage1_att = SelfAttentionModule_CoAM(spat_dims=spat_dims[:2], channel_list=self.stage2_cfg['NUM_CHANNELS'])
        if self.selfatt_config[1]:
            self.stage2_att = SelfAttentionModule_CoAM(spat_dims=spat_dims[:3], channel_list=self.stage3_cfg['NUM_CHANNELS'])
        if self.selfatt_config[2]:
            self.stage3_att = SelfAttentionModule_CoAM(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS'])
        if self.selfatt_config[3]:
            self.stage4_att = SelfAttentionModule_CoAM(spat_dims=[spat_dims[0]], channel_list=[self.stage4_cfg['NUM_CHANNELS'][0]])

        # ------------------------------------------------

        return

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def forward(self, x):

        x = x.cuda()
        
        if self.cfg.MODEL.EXTRA.USE_ATTENTION:
            if x[:,3:].shape[1] == 0:
                raise Exception("condition is empty, please check your dataloader")
            x_ = x[:,:3]
            cond_hm = x[:,3:]
        else:
            x_ = x

        x = self.conv1(x_)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # -------------------
        if self.cfg.MODEL.EXTRA.USE_ATTENTION and self.att_config[0]:
            x_list = self.stage1_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
    
        # -------------------
        if self.cfg.MODEL.EXTRA.USE_ATTENTION and self.att_config[1]:
                x_list = self.stage2_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # -------------------
        if self.cfg.MODEL.EXTRA.USE_ATTENTION and self.att_config[2]:
            x_list = self.stage3_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage4(x_list)

        # -------------------
        if self.cfg.MODEL.EXTRA.USE_ATTENTION and self.att_config[3]:
            y_list = self.stage4_att(y_list, cond_hm)
        # -------------------

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def _load_hrnet_coam(cfg, is_train, **kwargs) -> nn.Module:
    """
    Loads a HRNet with CoAM model.

    Args:
        cfg: the configuration file
        is_train: whether the model is in training mode

    Returns:
        the HRNet + CoAM model
    """
    model = HRNet_CoAM(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
