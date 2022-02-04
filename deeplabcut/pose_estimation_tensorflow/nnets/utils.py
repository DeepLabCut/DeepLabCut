# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.datasets import Batch
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.tpu import tpu_function


def wrapper(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


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
            batch_spec[Batch.pairwise_targets] = [
                batch_size,
                None,
                None,
                num_limbs * 2,
            ]
            batch_spec[Batch.pairwise_mask] = [
                batch_size,
                None,
                None,
                num_limbs * 2,
            ]
    return batch_spec


def make_2d_gaussian_kernel(sigma, size):
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    k = tf.range(-size // 2 + 1, size // 2 + 1)
    k = tf.cast(k ** 2, sigma.dtype)
    k = tf.nn.softmax(-k / (2 * (sigma ** 2)))
    return tf.einsum("i,j->ij", k, k)


def build_learning_rate(
    initial_lr,
    global_step,
    steps_per_epoch=None,
    lr_decay_type="exponential",
    decay_factor=0.97,
    decay_epochs=2.4,
    total_steps=None,
    warmup_epochs=5,
):
    """Build learning rate."""
    if lr_decay_type == "exponential":
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.compat.v1.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True
        )
    elif lr_decay_type == "cosine":
        assert total_steps is not None
        lr = (
            0.5
            * initial_lr
            * (1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
        )
    elif lr_decay_type == "constant":
        lr = initial_lr
    else:
        assert False, "Unknown lr_decay_type : %s" % lr_decay_type

    if warmup_epochs:
        tf.compat.v1.logging.info("Learning rate warmup_epochs: %d" % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_lr = (
            initial_lr
            * tf.cast(global_step, tf.float32)
            / tf.cast(warmup_steps, tf.float32)
        )
        lr = tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: warmup_lr,
            false_fn=lambda: lr,
        )

    return lr


def build_optimizer(
    learning_rate, optimizer_name="rmsprop", decay=0.9, epsilon=0.001, momentum=0.9
):
    """Build optimizer."""
    if optimizer_name == "sgd":
        tf.compat.v1.logging.info("Using SGD optimizer")
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=learning_rate
        )
    elif optimizer_name == "momentum":
        tf.compat.v1.logging.info("Using Momentum optimizer")
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum
        )
    elif optimizer_name == "rmsprop":
        tf.compat.v1.logging.info("Using RMSProp optimizer")
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate, decay, momentum, epsilon
        )
    else:
        tf.compat.v1.logging.fatal("Unknown optimizer:", optimizer_name)
    return optimizer


class TpuBatchNormalization(tf.compat.v1.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError("TpuBatchNormalization does not support fused=True.")
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    @staticmethod
    def _cross_replica_average(t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError(
                    "num_shards: %d mod shards_per_group: %d, should be 0"
                    % (num_shards, num_shards_per_group)
                )
            num_groups = num_shards // num_shards_per_group
            group_assignment = [
                [x for x in range(num_shards) if x // num_shards_per_group == y]
                for y in range(num_groups)
            ]
        return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype
        )

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims
        )

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
            num_shards_per_group = 1
        else:
            num_shards_per_group = max(8, num_shards // 8)
        tf.compat.v1.logging.info(
            "TpuBatchNormalization with num_shards_per_group %s", num_shards_per_group
        )
        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group
            )
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return group_mean, group_variance
        return shard_mean, shard_variance


class BatchNormalization(tf.compat.v1.layers.BatchNormalization):
    """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

    def __init__(self, name="tpu_batch_normalization", **kwargs):
        super(BatchNormalization, self).__init__(name=name, **kwargs)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(input=inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.compat.v1.div(inputs, keep_prob) * binary_tensor
    return output


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.compat.v1.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass
