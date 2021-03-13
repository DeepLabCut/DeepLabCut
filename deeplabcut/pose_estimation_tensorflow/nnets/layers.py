import tensorflow as tf
import tensorflow.contrib.slim as slim


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg['weight_decay']),
    ):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(
                input, num_outputs, kernel_size=[3, 3], stride=2, scope="block4"
            )
            return pred
