import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.models.heads.base import HEADS
from deeplabcut.pose_estimation_pytorch.models.modules.conv_block import BLOCKS
from deeplabcut.pose_estimation_pytorch.models.modules import BasicBlock, AdaptBlock
from .base import BaseHead

@HEADS.register_module
class HeatmapDEKRHead(BaseHead):

    def __init__(
            self,
            channels,
            num_blocks,
            dilation_rate,
            final_conv_kernel,
            block = BasicBlock,
            ):
        super().__init__()
        self.bn_momentum = 0.1
        self.inp_channels = channels[0]
        self.num_joints_with_center = channels[2] #Should account for the center being a joint
        self.final_conv_kernel = final_conv_kernel

        self.transition_heatmap = self._make_transition_for_head(
            self.inp_channels,
            channels[1],
        )
        self.head_heatmap = self._make_heatmap_head(
            block,
            num_blocks,
            channels[1],
            dilation_rate,
        )


    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, block, num_blocks, num_channels, dilation_rate):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            block,
            num_channels,
            num_channels,
            num_blocks,
            dilation=dilation_rate
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.num_joints_with_center,
            kernel_size=self.final_conv_kernel,
            stride=1,
            padding=1 if self.final_conv_kernel == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)
        
        return nn.ModuleList(heatmap_head_layers)
        
    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.bn_momentum),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        heatmap = self.head_heatmap[1](
            self.head_heatmap[0](
                self.transition_heatmap(x)
            )
        )

        return heatmap
    

@HEADS.register_module
class OffsetDEKRHead(BaseHead):

    def __init__(
            self,
            channels,
            num_offset_per_kpt,
            num_blocks,
            dilation_rate,
            final_conv_kernel,
            block = AdaptBlock,
    ):
        super().__init__()
        self.inp_channels = channels[0]
        self.num_joints = channels[2]
        self.num_joints_with_center = self.num_joints + 1

        self.bn_momentum = 0.1
        self.offset_perkpt = num_offset_per_kpt
        self.num_joints_without_center = self.num_joints
        self.offset_channels = self.offset_perkpt*self.num_joints_without_center
        assert(self.offset_channels == channels[1])

        self.num_blocks=num_blocks
        self.dilation_rate = dilation_rate
        self.final_conv_kernel = final_conv_kernel

        self.transition_offset = self._make_transition_for_head(
            self.inp_channels,
            self.offset_channels,
        )
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(
                block,
                num_blocks=num_blocks,
                num_channels_per_kpt=self.offset_perkpt,
                dilation_rate=self.dilation_rate
            )


    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.bn_momentum),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_separete_regression_head(
            self,
            block,
            num_blocks,
            num_channels_per_kpt,
            dilation_rate,
    ):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                block,
                num_channels_per_kpt,
                num_channels_per_kpt,
                num_blocks,
                dilation=dilation_rate
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=num_channels_per_kpt,
                out_channels=2,
                kernel_size=self.final_conv_kernel,
                stride=1,
                padding=1 if self.final_conv_kernel == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)
    
    def forward(self, x):
        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[:,j*self.offset_perkpt:(j+1)*self.offset_perkpt])))

        offset = torch.cat(final_offset, dim=1)

        return offset