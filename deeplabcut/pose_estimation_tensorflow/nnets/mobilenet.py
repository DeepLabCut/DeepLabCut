"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Also see our paper:
Pretraining boosts out-of-domain robustness for pose estimation
by Alexander Mathis, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis
https://arxiv.org/abs/1909.11229

Based on Slim implementation of mobilenets:
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
"""

import tensorflow as tf
import tf_slim as slim

from deeplabcut.pose_estimation_tensorflow.backbones import mobilenet_v2
from .base import BasePoseNet
from .factory import PoseNetFactory
from .layers import prediction_layer
from .utils import wrapper


networks = {
    "mobilenet_v2_1.0": (mobilenet_v2.mobilenet_base, mobilenet_v2.training_scope),
    "mobilenet_v2_0.75": (
        wrapper(
            mobilenet_v2.mobilenet_base,
            depth_multiplier=0.75,
            finegrain_classification_mode=True,
        ),
        mobilenet_v2.training_scope,
    ),
    "mobilenet_v2_0.5": (
        wrapper(
            mobilenet_v2.mobilenet_base,
            depth_multiplier=0.5,
            finegrain_classification_mode=True,
        ),
        mobilenet_v2.training_scope,
    ),
    "mobilenet_v2_0.35": (
        wrapper(
            mobilenet_v2.mobilenet_base,
            depth_multiplier=0.35,
            finegrain_classification_mode=True,
        ),
        mobilenet_v2.training_scope,
    ),
}


@PoseNetFactory.register("mobilenet")
class PoseMobileNet(BasePoseNet):
    def __init__(self, cfg):
        super(PoseMobileNet, self).__init__(cfg)

    def extract_features(self, inputs):
        net_fun, net_arg_scope = networks[self.cfg["net_type"]]
        im_centered = self.center_inputs(inputs)
        with slim.arg_scope(net_arg_scope()):
            net, end_points = net_fun(im_centered)

        return net, end_points

    def prediction_layers(
        self, features, end_points, scope="pose", reuse=None,
    ):
        out = super(PoseMobileNet, self).prediction_layers(features, scope, reuse,)
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if self.cfg["intermediate_supervision"]:
                out["part_pred_interm"] = prediction_layer(
                    self.cfg,
                    end_points[f"layer_{self.cfg['intermediate_supervision_layer']}"],
                    "intermediate_supervision",
                    self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                )
        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)
