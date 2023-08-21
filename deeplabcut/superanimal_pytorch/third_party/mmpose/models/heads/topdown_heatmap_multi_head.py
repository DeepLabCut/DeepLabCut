# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
    constant_init,
    normal_init,
)

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
import torch.nn.functional as F


@HEADS.register_module()
class TopdownHeatmapMultiHead(TopdownHeatmapBaseHead):
    def __init__(
        self,
        in_channels=27,
        use_conv_head=False,
        align_corners=False,
        loss_keypoint=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.loss = build_loss(loss_keypoint)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get("target_type", "GaussianHeatmap")
        self.align_corners = align_corners
        self.use_conv_head = use_conv_head
        if use_conv_head:
            self.keypoints_adp = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1)),
                nn.BatchNorm2d(in_channels),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(in_channels, in_channels, (3,3),padding=(1,1)),
                # nn.BatchNorm2d(in_channels),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(in_channels, in_channels, (3, 3)),
                # nn.BatchNorm2d(in_channels),
            )

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3
        losses["heatmap_loss"] = 0
        """
        for i in range(len(output)):
            feature = output['original'][i]
            target_resized = resize(target,
                                         size=(feature.shape[2], feature.shape[3]),
                                         mode='nearest')
            # print(target)
            # print(self.loss(feature, target_resized, target_weight))
            losses['heatmap_loss'] += self.loss(feature, target_resized, target_weight)
        losses['heatmap_loss'] = losses['heatmap_loss'] / len(output)
        """

        feature = output["original"][0]

        losses["heatmap_loss"] = self.loss(feature, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """
        if isinstance(output, dict):
            output = output["original"]

        accuracy = dict()
        accuracy["acc_pose"] = 0
        if self.target_type == "GaussianHeatmap":
            # for i in range(len(output)):
            #     feature = output[i]
            #     target_inter = resize(target, size=(feature.shape[2],
            #                                        feature.shape[3]),
            #                                  mode='nearest')
            _, avg_acc, _ = pose_pck_accuracy(
                output[0].detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0,
            )
            accuracy["acc_pose"] = float(avg_acc)
        return accuracy

    def forward(self, x):
        """Forward function."""
        if self.use_conv_head:
            feature = self.keypoints_adp(x[0])
            x = [feature] + list(x[1:])
        out = {}

        out["original"] = x

        return out

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if isinstance(output, dict):
            output = output["original"]
        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(), flip_pairs, target_type=self.target_type
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get("shift_heatmap", False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output[0].detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def init_weights(self):
        """Initialize model weights."""
        pass
        # for _, m in self.deconv_layers.named_modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         normal_init(m, std=0.001)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_init(m, 1)
        # for m in self.final_layer.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.001, bias=0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_init(m, 1)
