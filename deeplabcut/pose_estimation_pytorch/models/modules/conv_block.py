# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.ops as ops

from abc import ABC, abstractmethod

from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg

BLOCKS = Registry('blocks', build_func=build_from_cfg)


class BaseBlock(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self.bn_momentum = 0.1

    @abstractmethod
    def forward(self, x):
        pass
    
    def _init_weights(self, pretrained):
        if not pretrained:
            pass
        else:
            self.load_state_dict(torch.load(pretrained))

@BLOCKS.register_module
class BasicBlock(BaseBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=self.bn_momentum)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@BLOCKS.register_module
class Bottleneck(BaseBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, 
            downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=self.bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=self.bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BLOCKS.register_module
class AdaptBlock(BaseBlock):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
                                       [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
            padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
        self.bn = nn.BatchNorm2d(outplanes, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset-self.regular_matrix
        offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

        translation = self.translation_conv(x)
        offset[:,0::2,:,:] += translation[:,0:1,:,:]
        offset[:,1::2,:,:] += translation[:,1:2,:,:]
 
        out = self.adapt_conv(x, offset)
        out = self.bn(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

