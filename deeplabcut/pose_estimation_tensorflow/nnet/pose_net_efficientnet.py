"""
DeepLabCut 2.1.9 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
Effnet added by T. Biasi & AM

See our paper:

Pretraining boosts out-of-domain robustness for pose estimation
by Alexander Mathis, Thomas Biasi, Steffen Schneider, Mert Yuksekgonul, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 1859-1868
https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html
and preprint
https://arxiv.org/abs/1909.11229

Implements EffNet backbones with 1 deconv. layer and overall stride 8 for single and multi-animal (with PAF)
"""

import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import deeplabcut.pose_estimation_tensorflow.nnet.efficientnet_builder as eff
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses


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
        print("Getting specs", cfg["dataset_type"], cfg["num_limbs"], cfg["num_joints"])
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
            batch_spec[Batch.pairwise_targets] = [
                batch_size,
                None,
                None,
                cfg["num_limbs"] * 2,
            ]
            batch_spec[Batch.pairwise_mask] = [
                batch_size,
                None,
                None,
                cfg["num_limbs"] * 2,
            ]
    return batch_spec


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg
        if "use_batch_norm" not in self.cfg.keys():
            self.cfg["use_batch_norm"] = False
        if "use_drop_out" not in self.cfg.keys():
            self.cfg["use_drop_out"] = False

    def extract_features(self, inputs, use_batch_norm=False, use_drop_out=False):
        mean = tf.constant(
            self.cfg["mean_pixel"],
            dtype=tf.float32,
            shape=[1, 1, 1, 3],
            name="img_mean",
        )
        im_centered = inputs - mean
        im_centered /= tf.constant(eff.STDDEV_RGB, shape=[1, 1, 3])
        with tf.variable_scope("efficientnet"):
            eff_net_type = self.cfg["net_type"].replace("_", "-")
            net, end_points = eff.build_model_base(
                im_centered,
                eff_net_type,
                use_batch_norm=use_batch_norm,
                drop_out=use_drop_out,
            )
        return net, end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        out = {}
        with tf.variable_scope("pose", reuse=reuse):
            out["part_pred"] = prediction_layer(
                cfg,
                features,
                "part_pred",
                cfg["num_joints"] + cfg.get("num_idchannel", 0),
            )
            if cfg["location_refinement"]:
                out["locref"] = prediction_layer(
                    cfg, features, "locref_pred", cfg["num_joints"] * 2
                )
            if cfg["pairwise_predict"] and "multi-animal" not in cfg["dataset_type"]:
                out["pairwise_pred"] = prediction_layer(
                    cfg,
                    features,
                    "pairwise_pred",
                    cfg["num_joints"] * (cfg["num_joints"] - 1) * 2,
                )
            if (
                cfg["partaffinityfield_predict"]
                and "multi-animal" in cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    cfg, features, "pairwise_pred", cfg["num_limbs"] * 2
                )
            if cfg["intermediate_supervision"]:
                raise NotImplementedError(
                    "Intermediate supervision is currently disabled."
                )

        return out

    def get_net(self, inputs, use_batch_norm, use_drop_out):
        net, end_points = self.extract_features(inputs, use_batch_norm, use_drop_out)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(
            inputs, self.cfg["use_batch_norm"], self.cfg["use_drop_out"]
        )
        return self.add_inference_layers(heads)

    def inference(self, inputs):
        """ Direct TF inference on GPU.
        Added with: https://arxiv.org/abs/1909.11229
        """
        heads = self.get_net(inputs, False, False)
        locref = heads["locref"]
        probs = tf.sigmoid(heads["part_pred"])

        if self.cfg["batch_size"] == 1:
            # assuming batchsize 1 here!
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
                + offset * self.cfg["locref_stdev"]
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
                + offset * self.cfg["locref_stdev"]
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

        heads = self.get_net(
            batch[Batch.inputs], self.cfg["use_batch_norm"], self.cfg["use_drop_out"]
        )

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
            raise NotImplementedError("Intermediate supervision is currently disabled.")

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
