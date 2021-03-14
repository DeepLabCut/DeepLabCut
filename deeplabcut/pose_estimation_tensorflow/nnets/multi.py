"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import re
import tensorflow as tf
import tf_slim as slim
from tf_slim.nets import resnet_v1

import deeplabcut.pose_estimation_tensorflow.backbones.efficientnet_builder as eff
from deeplabcut.pose_estimation_tensorflow.nnets import conv_blocks
from deeplabcut.pose_estimation_tensorflow.backbones import mobilenet_v2, mobilenet
from .base import BasePoseNet
from .factory import PoseNetFactory
from .layers import prediction_layer
from .utils import wrapper


# Change the stride from 2 to 1 to get 16x downscaling instead of 32x.
mobilenet_v2.V2_DEF["spec"][14] = mobilenet.op(conv_blocks.expanded_conv, stride=1, num_outputs=160)


net_funcs = {
    "resnet_50": resnet_v1.resnet_v1_50,
    "resnet_101": resnet_v1.resnet_v1_101,
    "resnet_152": resnet_v1.resnet_v1_152,
    'mobilenet_v2_1.0': mobilenet_v2.mobilenet_base,
    'mobilenet_v2_0.75': wrapper(mobilenet_v2.mobilenet_base,
                                 depth_multiplier=0.75,
                                 final_endpoint="layer_19",
                                 finegrain_classification_mode=True),
    'mobilenet_v2_0.5': wrapper(mobilenet_v2.mobilenet_base,
                                depth_multiplier=0.5,
                                final_endpoint="layer_19",
                                finegrain_classification_mode=True),
    'mobilenet_v2_0.35': wrapper(mobilenet_v2.mobilenet_base,
                                 depth_multiplier=0.35,
                                 final_endpoint="layer_19",
                                 finegrain_classification_mode=True),
    'mobilenet_v2_0.1': wrapper(mobilenet_v2.mobilenet_base,
                                depth_multiplier=0.1,
                                final_endpoint="layer_19",
                                finegrain_classification_mode=True),
    'mobilenet_v2_0.35_10': wrapper(mobilenet_v2.mobilenet_base,
                                    depth_multiplier=0.35,
                                    final_endpoint="layer_10",
                                    finegrain_classification_mode=True),
    'mobilenet_v2_0.1_10':  wrapper(mobilenet_v2.mobilenet_base,
                                    depth_multiplier=0.1,
                                    final_endpoint="layer_10",
                                    finegrain_classification_mode=True)
}

#https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
parallel_layers = {
    "b0": "4",
    "b1": "7",
    "b2": "7",
    "b3": "7",
    "b4": "9",
    "b5": "12",
    "b6": "14",
    "b7": "17"
    }


@PoseNetFactory.register("multi")
class PoseMultiNet(BasePoseNet):
    def __init__(self, cfg):
        super(PoseMultiNet, self).__init__(cfg)
        if 'use_batch_norm' not in self.cfg.keys():
            self.cfg['use_batch_norm'] = False
        if 'use_drop_out' not in self.cfg.keys():
            self.cfg['use_drop_out'] = False

    def extract_features(self, inputs):
        im_centered = self.center_inputs(inputs)
        net_type = self.cfg['net_type']
        if 'resnet' in net_type:
            net_fun = net_funcs[net_type]
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(
                    im_centered, global_pool=False, output_stride=16, is_training=False
                )
        elif 'mobilenet' in net_type:
            net_fun = net_funcs[net_type]
            with slim.arg_scope(mobilenet_v2.training_scope()):
                net, end_points = net_fun(im_centered)
        elif 'efficientnet' in net_type:
            im_centered /= tf.constant(eff.STDDEV_RGB, shape=[1, 1, 3])
            net, end_points = eff.build_model_base(
                im_centered,
                net_type,
                use_batch_norm=self.cfg['use_batch_norm'],
                drop_out=self.cfg['use_drop_out']
            )
        else:
            raise ValueError(f"Unknown network of type {net_type}")
        return net, end_points

    def prediction_layers(
        self,
        features,
        end_points,
        input_shape,
        scope="pose",
        reuse=None,
    ):
        net_type = self.cfg['net_type']
        if "resnet" in net_type:
            num_layers = re.findall("resnet_([0-9]*)", net_type)[0]
            layer_name = "resnet_v1_{}/block{}/unit_{}/bottleneck_v1"
            mid_pt = layer_name.format(num_layers, 2, 3)
        elif "mobilenet" in net_type:
            mid_pt = "layer_7"
        elif "efficientnet" in net_type:
            mid_pt = f"block_{parallel_layers[net_type.split('-')[1]]}"
        else:
            raise ValueError(f"Unknown network of type {net_type}")

        final_dims = tf.math.ceil(
            tf.divide(input_shape[1:3], tf.convert_to_tensor(value=16))
        )
        interim_dims = tf.scalar_mul(2, final_dims)
        interim_dims = tf.cast(interim_dims, tf.int32)
        bank_3 = end_points[mid_pt]
        bank_3 = tf.image.resize(bank_3, interim_dims)

        with slim.arg_scope(
            [slim.conv2d],
            padding="SAME",
            normalizer_fn=None,
            weights_regularizer=tf.keras.regularizers.l2(0.5 * (self.cfg['weight_decay'])),
        ):
            with tf.compat.v1.variable_scope("decoder_filters"):
                bank_3 = slim.conv2d(bank_3, self.cfg['bank3'], 1, scope="decoder_parallel_1")

        with slim.arg_scope(
            [slim.conv2d_transpose],
            padding="SAME",
            normalizer_fn=None,
            weights_regularizer=tf.keras.regularizers.l2(0.5 * (self.cfg['weight_decay'])),
        ):
            with tf.compat.v1.variable_scope("upsampled_features"):
                upsampled_features = slim.conv2d_transpose(
                    features, self.cfg['bank5'], kernel_size=[3, 3], stride=2, scope="block4",
                )
        net = tf.concat([bank_3, upsampled_features], 3)
        out = super(PoseMultiNet, self).prediction_layers(
            net, scope, reuse,
        )
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if self.cfg['intermediate_supervision'] and "efficientnet" not in net_type:
                if "mobilenet" in net_type:
                    feat = end_points[f"layer_{self.cfg['intermediate_supervision_layer']}"]
                elif "resnet" in net_type:
                    layer_name = "resnet_v1_{}/block{}/unit_{}/bottleneck_v1"
                    num_layers = re.findall("resnet_([0-9]*)", self.cfg['net_type'])[0]
                    interm_name = layer_name.format(
                        num_layers, 3, self.cfg['intermediate_supervision_layer']
                    )
                    feat = end_points[interm_name]
                else:
                    return out
                pred_layer = out["part_pred_interm"] = prediction_layer(
                    self.cfg,
                    feat,
                    "intermediate_supervision",
                    self.cfg['num_joints'] + self.cfg.get("num_idchannel", 0),
                )
                out["part_pred_interm"] = pred_layer
        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points, tf.shape(input=inputs))
