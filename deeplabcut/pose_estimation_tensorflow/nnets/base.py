import abc
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.datasets import Batch
from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal
from .layers import prediction_layer
from .utils import make_2d_gaussian_kernel


class BasePoseNet(metaclass=abc.ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_features(self, inputs):
        ...

    @abc.abstractmethod
    def get_net(self, inputs):
        ...

    def train(self, batch):
        heads = self.get_net(batch[Batch.inputs])
        if self.cfg["weigh_part_predictions"]:
            part_score_weights = batch[Batch.part_score_weights]
        else:
            part_score_weights = 1.0

        def add_part_loss(pred_layer):
            return tf.compat.v1.losses.sigmoid_cross_entropy(
                batch[Batch.part_score_targets], heads[pred_layer], part_score_weights
            )

        loss = {"part_loss": add_part_loss("part_pred")}
        total_loss = loss["part_loss"]

        if (
            self.cfg["intermediate_supervision"]
            and "efficientnet" not in self.cfg["net_type"]
        ):
            loss["part_loss_interm"] = add_part_loss("part_pred_interm")
            total_loss += loss["part_loss_interm"]

        if self.cfg["location_refinement"]:
            locref_pred = heads["locref"]
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]
            loss_func = (
                tf.compat.v1.losses.huber_loss
                if self.cfg["locref_huber_loss"]
                else tf.compat.v1.losses.mean_squared_error
            )
            loss["locref_loss"] = self.cfg["locref_loss_weight"] * loss_func(
                locref_targets, locref_pred, locref_weights
            )
            total_loss += loss["locref_loss"]

        if self.cfg["pairwise_predict"] or self.cfg["partaffinityfield_predict"]:
            pairwise_pred = heads["pairwise_pred"]
            pairwise_targets = batch[Batch.pairwise_targets]
            pairwise_weights = batch[Batch.pairwise_mask]
            loss_func = (
                tf.compat.v1.losses.huber_loss
                if self.cfg["pairwise_huber_loss"]
                else tf.compat.v1.losses.mean_squared_error
            )
            loss["pairwise_loss"] = self.cfg["pairwise_loss_weight"] * loss_func(
                pairwise_targets, pairwise_pred, pairwise_weights
            )
            total_loss += loss["pairwise_loss"]

        loss["total_loss"] = total_loss
        return loss

    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_inference_layers(heads)

    def prediction_layers(
        self, features, scope="pose", reuse=None,
    ):
        out = {}
        n_joints = self.cfg["num_joints"]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out["part_pred"] = prediction_layer(
                self.cfg,
                features,
                "part_pred",
                n_joints + self.cfg.get("num_idchannel", 0),
            )
            if self.cfg["location_refinement"]:
                out["locref"] = prediction_layer(
                    self.cfg, features, "locref_pred", n_joints * 2,
                )
            if (
                self.cfg["pairwise_predict"]
                and "multi-animal" not in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg, features, "pairwise_pred", n_joints * (n_joints - 1) * 2,
                )
            if (
                self.cfg["partaffinityfield_predict"]
                and "multi-animal" in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg, features, "pairwise_pred", self.cfg["num_limbs"] * 2,
                )
        out["features"] = features
        return out

    def inference(self, inputs):
        """Direct TF inference on GPU.
        Added with: https://arxiv.org/abs/1909.11229
        """
        heads = self.get_net(inputs)
        locref = heads["locref"]
        probs = tf.sigmoid(heads["part_pred"])

        if self.cfg["batch_size"] == 1:
            probs = tf.squeeze(probs, axis=0)
            locref = tf.squeeze(locref, axis=0)
            l_shape = tf.shape(input=probs)
            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(input=probs, axis=0)
            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )
            maxloc = tf.reshape(maxloc, (1, -1))

            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2], dtype=tf.int64)), (1, -1)
            )
        else:
            l_shape = tf.shape(
                input=probs
            )  # batchsize times x times y times body parts
            locref = tf.reshape(
                locref, (l_shape[0], l_shape[1], l_shape[2], l_shape[3], 2)
            )
            # turn into x times y time bs * bpts
            locref = tf.transpose(a=locref, perm=[1, 2, 0, 3, 4])
            probs = tf.transpose(a=probs, perm=[1, 2, 0, 3])

            l_shape = tf.shape(input=probs)  # x times y times batch times body parts

            locref = tf.reshape(locref, (l_shape[0] * l_shape[1], -1, 2))
            probs = tf.reshape(probs, (l_shape[0] * l_shape[1], -1))
            maxloc = tf.argmax(input=probs, axis=0)
            loc = tf.unravel_index(
                maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))
            )  # tuple of max indices
            maxloc = tf.reshape(maxloc, (1, -1))
            joints = tf.reshape(
                tf.range(0, tf.cast(l_shape[2] * l_shape[3], dtype=tf.int64)), (1, -1)
            )

        # extract corresponding locref x and y as well as probability
        indices = tf.transpose(a=tf.concat([maxloc, joints], axis=0))
        offset = tf.gather_nd(locref, indices)
        offset = tf.gather(offset, [1, 0], axis=1)
        likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1, 1))

        pose = (
            self.cfg["stride"] * tf.cast(tf.transpose(a=loc), dtype=tf.float32)
            + self.cfg["stride"] * 0.5
            + offset * self.cfg["locref_stdev"]
        )
        pose = tf.concat([pose, likelihood], axis=1)
        return {"pose": pose}

    def add_inference_layers(self, heads):
        """initialized during inference"""
        prob = tf.sigmoid(heads["part_pred"])
        nms_radius = int(self.cfg.get("nmsradius", 5))

        # Filter predicted heatmaps with a 2D Gaussian kernel as in:
        # https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.pdf
        scmaps = tf.gather(prob, tf.range(self.cfg["num_joints"]), axis=3)
        kernel = make_2d_gaussian_kernel(
            sigma=self.cfg.get("sigma", 1), size=nms_radius * 2 + 1,
        )
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]

        kernel_sc = tf.tile(kernel, [1, 1, tf.shape(scmaps)[3], 1])
        scmaps = tf.nn.depthwise_conv2d(
            scmaps, kernel_sc, strides=[1, 1, 1, 1], padding="SAME",
        )
        peak_inds = predict_multianimal.find_local_peak_indices_maxpool_nms(
            scmaps, nms_radius, self.cfg.get("minconfidence", 0.01),
        )
        outputs = {"part_prob": prob, "peak_inds": peak_inds}
        if self.cfg["location_refinement"]:
            locref = heads["locref"]
            if self.cfg.get("locref_smooth", False):
                kernel_loc = tf.tile(kernel, [1, 1, tf.shape(locref)[3], 1])
                locref = tf.nn.depthwise_conv2d(
                    locref, kernel_loc, strides=[1, 1, 1, 1], padding="SAME",
                )
            outputs["locref"] = locref

        if self.cfg["pairwise_predict"] or self.cfg["partaffinityfield_predict"]:
            outputs["pairwise_pred"] = heads["pairwise_pred"]

        if "features" in heads:
            outputs["features"] = heads["features"]

        return outputs

    def center_inputs(self, inputs):
        mean = tf.constant(
            self.cfg["mean_pixel"],
            dtype=tf.float32,
            shape=[1, 1, 1, 3],
            name="img_mean",
        )
        return inputs - mean
