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
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
import deeplabcut.pose_estimation_tensorflow.nnet.efficientnet_builder as eff
from deeplabcut.pose_estimation_tensorflow.nnet import mobilenet_v2, mobilenet, conv_blocks
from deeplabcut.pose_estimation_tensorflow.nnet import losses

vers = (tf.__version__).split(".")
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf

# Change the stride from 2 to 1 to get 16x downscaling instead of 32x.
mobilenet_v2.V2_DEF["spec"][14] = mobilenet.op(conv_blocks.expanded_conv, stride=1, num_outputs=160)
def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

net_funcs = {
    "resnet_50": resnet_v1.resnet_v1_50,
    "resnet_101": resnet_v1.resnet_v1_101,
    "resnet_152": resnet_v1.resnet_v1_152,
    'mobilenet_v2_1.0': mobilenet_v2.mobilenet_base,
    'mobilenet_v2_0.75': wrapped_partial(mobilenet_v2.mobilenet_base,
                                        depth_multiplier=0.75,
                                        final_endpoint="layer_19",
                                        finegrain_classification_mode=True),
    'mobilenet_v2_0.5': wrapped_partial(mobilenet_v2.mobilenet_base,
                                        depth_multiplier=0.5,
                                        final_endpoint="layer_19",
                                        finegrain_classification_mode=True),
    'mobilenet_v2_0.35': wrapped_partial(mobilenet_v2.mobilenet_base,
                                        depth_multiplier=0.35,
                                        final_endpoint="layer_19",
                                        finegrain_classification_mode=True),
    'mobilenet_v2_0.1': wrapped_partial(mobilenet_v2.mobilenet_base,
                                        depth_multiplier=0.1,
                                        final_endpoint="layer_19",
                                        finegrain_classification_mode=True),
    'mobilenet_v2_0.35_10': wrapped_partial(mobilenet_v2.mobilenet_base,
                                            depth_multiplier=0.35,
                                            final_endpoint="layer_10",
                                            finegrain_classification_mode=True),
    'mobilenet_v2_0.1_10':  wrapped_partial(mobilenet_v2.mobilenet_base,
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


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg.weight_decay),
    ):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(
                input, num_outputs, kernel_size=[3, 3], stride=2, scope="block4"
            )
            return pred


def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    batch_spec = {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [
            batch_size,
            None,
            None,
            num_joints + cfg.get("num_idchannel", 0),
        ],
        Batch.part_score_weights: [
            batch_size,
            None,
            None,
            num_joints + cfg.get("num_idchannel", 0),
        ],
    }
    if cfg.location_refinement:
        batch_spec[Batch.locref_targets] = [batch_size, None, None, num_joints * 2]
        batch_spec[Batch.locref_mask] = [batch_size, None, None, num_joints * 2]
    if cfg.pairwise_predict:
        print("Getting specs", cfg.dataset_type, cfg.num_limbs, cfg.num_joints)
        if (
            "multi-animal" not in cfg.dataset_type
        ):  # this can be used for pairwise conditional
            batch_spec[Batch.pairwise_targets] = [
                batch_size,
                None,
                None,
                num_joints * (num_joints - 1) * 2,
            ]
            batch_spec[Batch.pairwise_mask] = [
                batch_size,
                None,
                None,
                num_joints * (num_joints - 1) * 2,
            ]
        else:  # train partaffinity fields
            batch_spec[Batch.pairwise_targets] = [
                batch_size,
                None,
                None,
                cfg.num_limbs * 2,
            ]
            batch_spec[Batch.pairwise_mask] = [
                batch_size,
                None,
                None,
                cfg.num_limbs * 2,
            ]
    return batch_spec


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg
        if 'use_batch_norm' not in self.cfg.keys():
            self.cfg.use_batch_norm = False
        if 'use_drop_out' not in self.cfg.keys():
            self.cfg.use_drop_out = False

    def extract_features(self, inputs):
        mean = tf.constant(
            self.cfg.mean_pixel, dtype=tf.float32, shape=[1, 1, 1, 3], name="img_mean"
        )
        im_centered = inputs - mean

        if 'resnet' in self.cfg.net_type:
            # The next part of the code depends upon which tensorflow version you have.
            vers = tf.__version__
            vers = vers.split(
                "."
            )  # Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44

            net_fun = net_funcs[self.cfg.net_type]
            if int(vers[0]) == 1 and int(vers[1]) < 4:  # check if lower than version 1.4.
                with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                    net, end_points = net_fun(
                        im_centered, global_pool=False, output_stride=16
                    )
            else:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    net, end_points = net_fun(
                        im_centered, global_pool=False, output_stride=16, is_training=False
                    )
        elif 'mobilenet' in self.cfg.net_type:
            net_fun = net_funcs[self.cfg.net_type]
            with slim.arg_scope(mobilenet_v2.training_scope()):
                net, end_points = net_fun(im_centered)
        elif 'efficientnet' in self.cfg.net_type:
            im_centered /= tf.constant(eff.STDDEV_RGB, shape=[1, 1, 3])
            net, end_points = eff.build_model_base(im_centered,
                                                   self.cfg.net_type,
                                                   use_batch_norm=self.cfg.use_batch_norm,
                                                   drop_out=self.cfg.use_drop_out)
        return net, end_points

    def prediction_layers(
        self,
        features,
        end_points,
        input_shape,
        reuse=None,
        is_training=False,
        scope="pose",
    ):
        cfg = self.cfg
        if "resnet" in cfg.net_type:
            num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
            layer_name = (
                "resnet_v1_{}".format(num_layers) + "/block{}/unit_{}/bottleneck_v1"
            )
            mid_pt = layer_name.format(2,3)
        elif "mobilenet" in cfg.net_type:
            mid_pt = "layer_7"
        elif "efficientnet" in cfg.net_type:
            mid_pt = "block_"+parallel_layers[cfg.net_type.split('-')[1]]

        final_dims = tf.ceil(
            tf.divide(input_shape[1:3], tf.convert_to_tensor(16))
        )
        interim_dims = tf.scalar_mul(2, final_dims)
        interim_dims = tf.cast(interim_dims, tf.int32)
        bank_3 = end_points[mid_pt]
        bank_3 = tf.image.resize_images(bank_3, interim_dims)

        with slim.arg_scope(
            [slim.conv2d],
            padding="SAME",
            normalizer_fn=None,
            weights_regularizer=slim.l2_regularizer(cfg.weight_decay),
        ):
            with tf.variable_scope("decoder_filters"):
                bank_3 = slim.conv2d(bank_3, cfg.bank3, 1, scope="decoder_parallel_1")

        with slim.arg_scope(
            [slim.conv2d_transpose],
            padding="SAME",
            normalizer_fn=None,
            weights_regularizer=slim.l2_regularizer(cfg.weight_decay),
        ):
            with tf.variable_scope("upsampled_features"):
                upsampled_features = slim.conv2d_transpose(
                    features, cfg.bank5, kernel_size=[3, 3], stride=2, scope="block4"
                )
        net = tf.concat([bank_3, upsampled_features], 3)

        out = {}
        with tf.variable_scope(scope, reuse=reuse):
            out["part_pred"] = prediction_layer(
                cfg, net, "part_pred", cfg.num_joints + cfg.get("num_idchannel", 0)
            )
            if cfg.location_refinement:
                out["locref"] = prediction_layer(
                    cfg, net, "locref_pred", cfg.num_joints * 2
                )
            if cfg.pairwise_predict and "multi-animal" not in cfg.dataset_type:
                out["pairwise_pred"] = prediction_layer(
                    cfg, net, "pairwise_pred", cfg.num_joints * (cfg.num_joints - 1) * 2
                )
            if cfg.partaffinityfield_predict and "multi-animal" in cfg.dataset_type:
                out["pairwise_pred"] = prediction_layer(
                    cfg, net, "pairwise_pred", cfg.num_limbs * 2
                )

            if cfg.intermediate_supervision and "efficientnet" not in cfg.net_type:
                if "mobilenet" in cfg.net_type:
                    out["part_pred_interm"] = prediction_layer(
                        cfg,
                        end_points["layer_" + str(cfg["intermediate_supervision_layer"])],
                        "intermediate_supervision",
                        cfg.num_joints,
                    )
                elif "resnet" in cfg.net_type:
                    interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                    block_interm_out = end_points[interm_name]
                    out["part_pred_interm"] = prediction_layer(
                        cfg,
                        block_interm_out,
                        "intermediate_supervision",
                        cfg.num_joints + cfg.get("num_idchannel", 0),
                    )

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        # return self.prediction_layers(net, end_points)
        return self.prediction_layers(net, end_points, tf.shape(inputs))

    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_inference_layers(heads)

    def add_inference_layers(self, heads):
        """ initialized during inference """
        prob = tf.sigmoid(heads["part_pred"])
        outputs = {"part_prob": prob}
        if self.cfg.location_refinement:
            outputs["locref"] = heads["locref"]
        if self.cfg.pairwise_predict or self.cfg.partaffinityfield_predict:
            outputs["pairwise_pred"] = heads["pairwise_pred"]
        return outputs

    def train(self, batch):
        cfg = self.cfg

        heads = self.get_net(batch[Batch.inputs])
        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = (
            batch[Batch.part_score_weights] if weigh_part_predictions else 1.0
        )

        def add_part_loss(pred_layer):
            return TF.losses.sigmoid_cross_entropy(
                batch[Batch.part_score_targets], heads[pred_layer], part_score_weights
            )

        loss = {}
        loss["part_loss"] = add_part_loss("part_pred")
        total_loss = loss["part_loss"]
        if cfg.intermediate_supervision and "efficientnet" not in cfg.net_type:
            loss["part_loss_interm"] = add_part_loss("part_pred_interm")
            total_loss = total_loss + loss["part_loss_interm"]

        if cfg.location_refinement:
            locref_pred = heads["locref"]
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]
            loss_func = (
                losses.huber_loss
                if cfg.locref_huber_loss
                else tf.losses.mean_squared_error
            )
            loss["locref_loss"] = cfg.locref_loss_weight * loss_func(
                locref_targets, locref_pred, locref_weights
            )
            total_loss = total_loss + loss["locref_loss"]

        if cfg.pairwise_predict or cfg.partaffinityfield_predict:
            "setting pw bodypart loss..."
            pairwise_pred = heads["pairwise_pred"]
            pairwise_targets = batch[Batch.pairwise_targets]
            pairwise_weights = batch[Batch.pairwise_mask]

            loss_func = (
                losses.huber_loss
                if cfg.pairwise_huber_loss
                else tf.losses.mean_squared_error
            )
            loss["pairwise_loss"] = cfg.pairwise_loss_weight * loss_func(
                pairwise_targets, pairwise_pred, pairwise_weights
            )
            total_loss = total_loss + loss["pairwise_loss"]

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss["total_loss"] = total_loss
        return loss
