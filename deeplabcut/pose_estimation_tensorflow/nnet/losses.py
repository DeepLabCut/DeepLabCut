'''
Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

import tensorflow.contrib.losses as tf_losses


def huber_loss(labels, predictions, weight=1.0, k=1.0, scope=None):
    """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss
      tensor: tensor to regularize.
      k: value of k in the huber loss
      scope: Optional scope for op_scope.

    Huber loss:
    f(x) = if |x| <= k:
              0.5 * x^2
           else:
              k * |x| - 0.5 * k^2

    Returns:
      the L1 loss op.

    http://concise-bio.readthedocs.io/en/latest/_modules/concise/tf_helper.html
    """
    with ops.name_scope(scope, "absolute_difference",
                        [predictions, labels]) as scope:
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        if weight is None:
            raise ValueError("`weight` cannot be None")
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        diff = math_ops.subtract(predictions, labels)
        abs_diff = tf.abs(diff)
        losses = tf.where(abs_diff < k,
                          0.5 * tf.square(diff),
                          k * abs_diff - 0.5 * k ** 2)
        return tf.losses.compute_weighted_loss(losses, weight)
