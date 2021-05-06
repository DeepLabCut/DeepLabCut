"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

See our paper:

Pretraining boosts out-of-domain robustness for pose estimation
by Alexander Mathis, Thomas Biasi, Steffen Schneider, Mert Yuksekgonul, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 1859-1868
https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html
and preprint
https://arxiv.org/abs/1909.11229

Based on Slim implementation of mobilenets:
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

Implements MobileNetv2 backbone with 1 deconv. layer and overall stride 8 for single and multi-animal (with PAF)
"""

import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

from deeplabcut.pose_estimation_tensorflow.nnet import mobilenet_v2
from . import losses
from ..dataset.pose_dataset import Batch


def wrapper(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


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


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg["weight_decay"]),
    ):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(
                input, num_outputs, kernel_size=[3, 3], stride=2, scope="block4"
            )
            return pred


def get_batch_spec(cfg):
    num_joints = cfg["num_joints"]
    num_limbs = cfg["num_limbs"]
    batch_size = cfg["batch_size"]
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
    if cfg["location_refinement"]:
        batch_spec[Batch.locref_targets] = [batch_size, None, None, num_joints * 2]
        batch_spec[Batch.locref_mask] = [batch_size, None, None, num_joints * 2]
    if cfg["pairwise_predict"]:
        print("Getting specs", cfg["dataset_type"], num_limbs, num_joints)
        if (
            "multi-animal" not in cfg["dataset_type"]
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
            batch_spec[Batch.pairwise_targets] = [batch_size, None, None, num_limbs * 2]
            batch_spec[Batch.pairwise_mask] = [batch_size, None, None, num_limbs * 2]
    return batch_spec


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, inputs):
        net_fun, net_arg_scope = networks[self.cfg["net_type"]]
        mean = tf.constant(
            self.cfg["mean_pixel"],
            dtype=tf.float32,
            shape=[1, 1, 1, 3],
            name="img_mean",
        )
        im_centered = inputs - mean
        with slim.arg_scope(net_arg_scope()):
            net, end_points = net_fun(im_centered)

        return net, end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg
        num_joints = cfg["num_joints"]

        out = {}
        with tf.variable_scope("pose", reuse=reuse):
            out["part_pred"] = prediction_layer(cfg, features, "part_pred", num_joints)
            if cfg["location_refinement"]:
                out["locref"] = prediction_layer(
                    cfg, features, "locref_pred", num_joints * 2
                )
            if cfg["pairwise_predict"] and "multi-animal" not in cfg["dataset_type"]:
                out["pairwise_pred"] = prediction_layer(
                    cfg, features, "pairwise_pred", num_joints * (num_joints - 1) * 2
                )
            if (
                cfg["partaffinityfield_predict"]
                and "multi-animal" in cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    cfg, features, "pairwise_pred", cfg["num_limbs"] * 2
                )
            if cfg["intermediate_supervision"]:
                # print(end_points.keys()) >> to see what else is available.
                out["part_pred_interm"] = prediction_layer(
                    cfg,
                    end_points["layer_" + str(cfg["intermediate_supervision_layer"])],
                    "intermediate_supervision",
                    num_joints,
                )

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_inference_layers(heads)

    def inference(self, inputs):
        """ Direct TF inference on GPU. Added with: https://arxiv.org/abs/1909.11229"""
        cfg = self.cfg
        heads = self.get_net(inputs)
        locref = heads["locref"]
        probs = tf.sigmoid(heads["part_pred"])

        if cfg["batch_size"] == 1:
            probs = tf.squeeze(probs, axis=0)
            locref = tf.squeeze(locref, axis=0)
            l_shape = tf.shape(probs)

            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(probs, axis=0)

            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )
            maxloc = tf.reshape(maxloc, (1, -1))

            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2], dtype=tf.int64)), (1, -1)
            )
            indices = tf.transpose(tf.concat([maxloc, joints], axis=0))

            offset = tf.gather_nd(locref, indices)
            offset = tf.gather(offset, [1, 0], axis=1)
            likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1, 1))

            pose = (
                self.cfg["stride"] * tf.cast(tf.transpose(loc), dtype=tf.float32)
                + self.cfg["stride"] * 0.5
                + offset * cfg["locref_stdev"]
            )
            pose = tf.concat([pose, likelihood], axis=1)

            return {"pose": pose}
        else:
            # probs = tf.squeeze(probs, axis=0)
            l_shape = tf.shape(probs)  # batchsize times x times y times body parts
            # locref = locref*cfg.locref_stdev
            locref = tf.reshape(
                locref, (l_shape[0], l_shape[1], l_shape[2], l_shape[3], 2)
            )
            # turn into x times y time bs * bpts
            locref = tf.transpose(locref, [1, 2, 0, 3, 4])
            probs = tf.transpose(probs, [1, 2, 0, 3])

            # print(locref.get_shape().as_list())
            # print(probs.get_shape().as_list())
            l_shape = tf.shape(probs)  # x times y times batch times body parts

            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(probs, axis=0)
            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )  # tuple of max indices

            maxloc = tf.reshape(maxloc, (1, -1))
            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2] * l_shape[3], dtype=tf.int64)), (1, -1)
            )
            indices = tf.transpose(tf.concat([maxloc, joints], axis=0))

            # extract corresponding locref x and y as well as probability
            offset = tf.gather_nd(locref, indices)
            offset = tf.gather(offset, [1, 0], axis=1)
            likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1, 1))

            pose = (
                self.cfg["stride"] * tf.cast(tf.transpose(loc), dtype=tf.float32)
                + self.cfg["stride"] * 0.5
                + offset * cfg["locref_stdev"]
            )
            pose = tf.concat([pose, likelihood], axis=1)
            return {"pose": pose}

    def add_inference_layers(self, heads):
        """ initialized during inference """
        prob = tf.sigmoid(heads["part_pred"])
        outputs = {"part_prob": prob}
        if self.cfg["location_refinement"]:
            outputs["locref"] = heads["locref"]
        if self.cfg["pairwise_predict"] or self.cfg["partaffinityfield_predict"]:
            outputs["pairwise_pred"] = heads["pairwise_pred"]
        return outputs

    def train(self, batch):
        cfg = self.cfg

        heads = self.get_net(batch[Batch.inputs])

        weigh_part_predictions = cfg["weigh_part_predictions"]
        part_score_weights = (
            batch[Batch.part_score_weights] if weigh_part_predictions else 1.0
        )

        def add_part_loss(pred_layer):
            return tf.losses.sigmoid_cross_entropy(
                batch[Batch.part_score_targets], heads[pred_layer], part_score_weights
            )

        loss = {}
        loss["part_loss"] = add_part_loss("part_pred")
        total_loss = loss["part_loss"]
        if cfg["intermediate_supervision"]:
            loss["part_loss_interm"] = add_part_loss("part_pred_interm")
            total_loss = total_loss + loss["part_loss_interm"]

        if cfg["location_refinement"]:
            locref_pred = heads["locref"]
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_func = (
                losses.huber_loss
                if cfg["locref_huber_loss"]
                else tf.losses.mean_squared_error
            )
            loss["locref_loss"] = cfg["locref_loss_weight"] * loss_func(
                locref_targets, locref_pred, locref_weights
            )
            total_loss = total_loss + loss["locref_loss"]

        if cfg["pairwise_predict"] or cfg["partaffinityfield_predict"]:
            # setting pairwise bodypart loss
            pairwise_pred = heads["pairwise_pred"]
            pairwise_targets = batch[Batch.pairwise_targets]
            pairwise_weights = batch[Batch.pairwise_mask]

            loss_func = (
                losses.huber_loss
                if cfg["pairwise_huber_loss"]
                else tf.losses.mean_squared_error
            )
            loss["pairwise_loss"] = cfg["pairwise_loss_weight"] * loss_func(
                pairwise_targets, pairwise_pred, pairwise_weights
            )
            total_loss = total_loss + loss["pairwise_loss"]

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss["total_loss"] = total_loss
        return loss
