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

import deeplabcut.pose_estimation_tensorflow.backbones.efficientnet_builder as eff
from deeplabcut.pose_estimation_tensorflow.nnets import conv_blocks
from deeplabcut.pose_estimation_tensorflow.backbones import mobilenet_v2, mobilenet
from .base import BasePoseNet
from .factory import PoseNetFactory
from .layers import prediction_layer_stage
from .utils import wrapper


# Change the stride from 2 to 1 to get 16x downscaling instead of 32x.
mobilenet_v2.V2_DEF["spec"][14] = mobilenet.op(
    conv_blocks.expanded_conv, stride=1, num_outputs=160
)


net_funcs = {
    "resnet_50": resnet_v1.resnet_v1_50,
    "resnet_101": resnet_v1.resnet_v1_101,
    "resnet_152": resnet_v1.resnet_v1_152,
    "mobilenet_v2_1.0": mobilenet_v2.mobilenet_base,
    "mobilenet_v2_0.75": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.75,
        final_endpoint="layer_19",
        finegrain_classification_mode=True,
    ),
    "mobilenet_v2_0.5": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.5,
        final_endpoint="layer_19",
        finegrain_classification_mode=True,
    ),
    "mobilenet_v2_0.35": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.35,
        final_endpoint="layer_19",
        finegrain_classification_mode=True,
    ),
    "mobilenet_v2_0.1": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.1,
        final_endpoint="layer_19",
        finegrain_classification_mode=True,
    ),
    "mobilenet_v2_0.35_10": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.35,
        final_endpoint="layer_10",
        finegrain_classification_mode=True,
    ),
    "mobilenet_v2_0.1_10": wrapper(
        mobilenet_v2.mobilenet_base,
        depth_multiplier=0.1,
        final_endpoint="layer_10",
        finegrain_classification_mode=True,
    ),
}

# https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
parallel_layers = {
    "b0": "4",
    "b1": "7",
    "b2": "7",
    "b3": "7",
    "b4": "9",
    "b5": "12",
    "b6": "14",
    "b7": "17",
}


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg["weight_decay"]),
    ):
        with tf.compat.v1.variable_scope(name):
            pred = slim.conv2d_transpose(
                input, num_outputs, kernel_size=[3, 3], stride=2,
            )
            return pred


@PoseNetFactory.register("multi")
class PoseMultiNet(BasePoseNet):
    def __init__(self, cfg):
        super(PoseMultiNet, self).__init__(cfg)
        multi_stage = self.cfg.get("multi_stage", False)
        # Multi stage is currently only implemented for resnets
        self.cfg["multi_stage"] = multi_stage and "resnet" in self.cfg["net_type"]

    def extract_features(self, inputs):
        im_centered = self.center_inputs(inputs)
        net_type = self.cfg["net_type"]
        if "resnet" in net_type:
            net_fun = net_funcs[net_type]
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(
                    im_centered, global_pool=False, output_stride=16, is_training=False
                )
        elif "mobilenet" in net_type:
            net_fun = net_funcs[net_type]
            with slim.arg_scope(mobilenet_v2.training_scope()):
                net, end_points = net_fun(im_centered)
        elif "efficientnet" in net_type:
            if "use_batch_norm" not in self.cfg.keys():
                self.cfg["use_batch_norm"] = False
            if "use_drop_out" not in self.cfg.keys():
                self.cfg["use_drop_out"] = False

            im_centered /= tf.constant(eff.STDDEV_RGB, shape=[1, 1, 3])
            net, end_points = eff.build_model_base(
                im_centered,
                net_type,
                use_batch_norm=self.cfg["use_batch_norm"],
                drop_out=self.cfg["use_drop_out"],
            )
        else:
            raise ValueError(f"Unknown network of type {net_type}")
        return net, end_points

    def prediction_layers(
        self, features, end_points, input_shape, scope="pose", reuse=None,
    ):
        net_type = self.cfg["net_type"]
        if self.cfg["multi_stage"]:  # MuNet! (multi_stage decoder + multi_fusion)
            # Defining multi_fusion backbone
            num_layers = re.findall("resnet_([0-9]*)", net_type)[0]
            layer_name = (
                "resnet_v1_{}".format(num_layers) + "/block{}/unit_{}/bottleneck_v1"
            )
            mid_pt_block1 = layer_name.format(1, 3)
            mid_pt_block2 = layer_name.format(2, 3)

            final_dims = tf.math.ceil(
                tf.divide(input_shape[1:3], tf.convert_to_tensor(16))
            )

            interim_dims_s8 = tf.scalar_mul(2, final_dims)
            interim_dims_s8 = tf.cast(interim_dims_s8, tf.int32)
            interim_dims_s4 = tf.scalar_mul(2, interim_dims_s8)
            interim_dims_s4 = tf.cast(interim_dims_s4, tf.int32)

            bank_1 = end_points[mid_pt_block1]
            bank_2 = end_points[mid_pt_block2]

            bank_2_s8 = tf.compat.v1.image.resize_images(bank_2, interim_dims_s8)
            bank_1_s4 = tf.compat.v1.image.resize_images(bank_1, interim_dims_s4)

            with slim.arg_scope(
                [slim.conv2d],
                padding="SAME",
                normalizer_fn=slim.layers.batch_norm,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(self.cfg["weight_decay"]),
            ):
                with tf.compat.v1.variable_scope("decoder_filters"):
                    bank_2_s16 = slim.conv2d(
                        bank_2_s8,
                        512,
                        kernel_size=[3, 3],
                        stride=2,
                        scope="decoder_parallel_1",
                    )
                    bank_2_s16 = slim.conv2d(
                        bank_2_s16,
                        128,
                        kernel_size=[1, 1],
                        stride=1,
                        scope="decoder_parallel_2",
                    )

                    bank_1_s8 = slim.conv2d(
                        bank_1_s4,
                        256,
                        kernel_size=[3, 3],
                        stride=2,
                        scope="decoder_parallel_3",
                    )
                    bank_1_s16 = slim.conv2d(
                        bank_1_s8,
                        256,
                        kernel_size=[3, 3],
                        stride=2,
                        scope="decoder_parallel_4",
                    )
                    bank_1_s16 = slim.conv2d(
                        bank_1_s16,
                        128,
                        kernel_size=[1, 1],
                        stride=1,
                        scope="decoder_parallel_5",
                    )

            with slim.arg_scope(
                [slim.conv2d_transpose],
                padding="SAME",
                normalizer_fn=None,
                weights_regularizer=slim.l2_regularizer(self.cfg["weight_decay"]),
            ):
                with tf.compat.v1.variable_scope("upsampled_features"):

                    concat_3_s16 = tf.concat([bank_1_s16, bank_2_s16, features], 3)

                    if self.cfg["stride"] == 8:
                        net = concat_3_s16

                    elif self.cfg["stride"] == 4:
                        upsampled_features_2x = slim.conv2d_transpose(
                            concat_3_s16,
                            self.cfg.get("bank3", 128),
                            kernel_size=[3, 3],
                            stride=2,
                            scope="block3",
                        )
                        net = upsampled_features_2x

                    elif self.cfg["stride"] == 2:
                        upsampled_features_2x = slim.conv2d_transpose(
                            concat_3_s16,
                            self.cfg.get("bank3", 128),
                            kernel_size=[3, 3],
                            stride=2,
                            scope="block3",
                        )
                        upsampled_features_4x = slim.conv2d_transpose(
                            upsampled_features_2x,
                            self.cfg.get("bank5", 128),
                            kernel_size=[3, 3],
                            stride=2,
                            scope="block4",
                        )
                        net = upsampled_features_4x

            out = {}
            # Attaching multi-stage decoder
            with tf.compat.v1.variable_scope(scope, reuse=reuse):
                stage1_hm_out = prediction_layer(
                    self.cfg,
                    net,
                    "part_pred_s1",
                    self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                )

                if self.cfg["location_refinement"]:
                    out["locref"] = prediction_layer(
                        self.cfg, net, "locref_pred", self.cfg["num_joints"] * 2
                    )
                if (
                    self.cfg["pairwise_predict"]
                    and "multi-animal" not in self.cfg["dataset_type"]
                ):
                    out["pairwise_pred"] = prediction_layer(
                        self.cfg,
                        net,
                        "pairwise_pred",
                        self.cfg["num_joints"] * (self.cfg["num_joints"] - 1) * 2,
                    )
                if (
                    self.cfg["partaffinityfield_predict"]
                    and "multi-animal" in self.cfg["dataset_type"]
                ):
                    feature = slim.conv2d_transpose(
                        net, self.cfg.get("bank3", 128), kernel_size=[3, 3], stride=2
                    )

                    stage1_paf_out = prediction_layer(
                        self.cfg, net, "pairwise_pred_s1", self.cfg["num_limbs"] * 2
                    )

                    stage2_in = tf.concat([stage1_hm_out, stage1_paf_out, feature], 3)
                    stage_input = stage2_in
                    stage_paf_output = stage1_paf_out
                    stage_hm_output = stage1_hm_out

                    for i in range(2, 5):
                        pre_stage_paf_output = stage_paf_output
                        pre_stage_hm_output = stage_hm_output

                        stage_paf_output = prediction_layer_stage(
                            self.cfg,
                            stage_input,
                            f"pairwise_pred_s{i}",
                            self.cfg["num_limbs"] * 2,
                        )

                        stage_hm_output = prediction_layer_stage(
                            self.cfg,
                            stage_input,
                            f"part_pred_s{i}",
                            self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                        )

                        if i > 2:
                            # stage_paf_output = stage_paf_output + pre_stage_paf_output
                            stage_hm_output = stage_hm_output + pre_stage_hm_output

                        stage_input = tf.concat(
                            [stage_hm_output, stage_paf_output, feature], 3
                        )

                    out["part_pred"] = prediction_layer_stage(
                        self.cfg,
                        stage_input,
                        "part_pred",
                        self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                    )

                    out["pairwise_pred"] = prediction_layer_stage(
                        self.cfg,
                        stage_input,
                        "pairwise_pred",
                        self.cfg["num_limbs"] * 2,
                    )

                if self.cfg["intermediate_supervision"]:
                    interm_name = layer_name.format(
                        3, self.cfg["intermediate_supervision_layer"]
                    )
                    block_interm_out = end_points[interm_name]
                    out["part_pred_interm"] = prediction_layer(
                        self.cfg,
                        block_interm_out,
                        "intermediate_supervision",
                        self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                    )

        else:  # dual fusion net (for stride 4 experiments)
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
                weights_regularizer=tf.keras.regularizers.l2(
                    0.5 * (self.cfg["weight_decay"])
                ),
            ):
                with tf.compat.v1.variable_scope("decoder_filters"):
                    bank_3 = slim.conv2d(
                        bank_3,
                        self.cfg.get("bank3", 128),
                        1,
                        scope="decoder_parallel_1",
                    )

            with slim.arg_scope(
                [slim.conv2d_transpose],
                padding="SAME",
                normalizer_fn=None,
                weights_regularizer=tf.keras.regularizers.l2(
                    0.5 * (self.cfg["weight_decay"])
                ),
            ):
                with tf.compat.v1.variable_scope("upsampled_features"):
                    upsampled_features = slim.conv2d_transpose(
                        features,
                        self.cfg.get("bank5", 128),
                        kernel_size=[3, 3],
                        stride=2,
                        scope="block4",
                    )
            net = tf.concat([bank_3, upsampled_features], 3)
            out = super(PoseMultiNet, self).prediction_layers(net, scope, reuse,)
            with tf.compat.v1.variable_scope(scope, reuse=reuse):
                if (
                    self.cfg["intermediate_supervision"]
                    and "efficientnet" not in net_type
                ):
                    if "mobilenet" in net_type:
                        feat = end_points[
                            f"layer_{self.cfg['intermediate_supervision_layer']}"
                        ]
                    elif "resnet" in net_type:
                        layer_name = "resnet_v1_{}/block{}/unit_{}/bottleneck_v1"
                        num_layers = re.findall("resnet_([0-9]*)", net_type)[0]
                        interm_name = layer_name.format(
                            num_layers, 3, self.cfg["intermediate_supervision_layer"]
                        )
                        feat = end_points[interm_name]
                    else:
                        return out
                    pred_layer = out["part_pred_interm"] = prediction_layer(
                        self.cfg,
                        feat,
                        "intermediate_supervision",
                        self.cfg["num_joints"] + self.cfg.get("num_idchannel", 0),
                    )
                    out["part_pred_interm"] = pred_layer
        out["features"] = features
        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points, tf.shape(input=inputs))
