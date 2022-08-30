"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""
import re
import tensorflow as tf
import tf_slim as slim
from tf_slim.nets import resnet_v1

from .base import BasePoseNet
from .factory import PoseNetFactory
from .layers import prediction_layer


net_funcs = {
    "resnet_50": resnet_v1.resnet_v1_50,
    "resnet_101": resnet_v1.resnet_v1_101,
    "resnet_152": resnet_v1.resnet_v1_152,
}


@PoseNetFactory.register("resnet")
class PoseResnet(BasePoseNet):
    def __init__(self, cfg):
        super(PoseResnet, self).__init__(cfg)

    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg["net_type"]]
        im_centered = self.center_inputs(inputs)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = net_fun(
                im_centered, global_pool=False, output_stride=16, is_training=False,
            )
        return net, end_points

    def prediction_layers(
        self, features, end_points, scope="pose", reuse=None,
    ):
        out = super(PoseResnet, self).prediction_layers(features, scope, reuse,)
        out["features"] = features
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if self.cfg["intermediate_supervision"]:
                layer_name = "resnet_v1_{}/block{}/unit_{}/bottleneck_v1"
                num_layers = re.findall("resnet_([0-9]*)", self.cfg["net_type"])[0]
                interm_name = layer_name.format(
                    num_layers, 3, self.cfg["intermediate_supervision_layer"]
                )
                block_interm_out = end_points[interm_name]
                out["part_pred_interm"] = prediction_layer(
                    self.cfg,
                    block_interm_out,
                    "intermediate_supervision",
                    self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                )
        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)
