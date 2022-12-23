#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import tensorflow as tf
import tf_slim as slim


# FIXME Fix wrong scope with Keras layers
# def prediction_layer(cfg, input, name, num_outputs):
#     with tf.compat.v1.variable_scope(name):
#         layer = tf.keras.layers.Conv2DTranspose(
#             filters=num_outputs,
#             kernel_size=(3, 3),
#             strides=2,
#             padding="same",
#             kernel_regularizer=tf.keras.regularizers.l2(0.5 * (cfg['weight_decay'])),
#             name=name,
#             dtype=input.dtype.base_dtype,
#         )
#         return layer(input)


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
                input, num_outputs, kernel_size=[3, 3], stride=2, scope="block4"
            )
            return pred


### New DLCNet Addition: multi-stage decoder
# def prediction_layer_stage(cfg, input, name, num_outputs):
#     with tf.compat.v1.variable_scope(name):
#         layer = tf.keras.layers.Conv2D(
#             filters=num_outputs,
#             kernel_size=(3, 3),
#             strides=1,
#             padding="same",
#             kernel_regularizer=tf.keras.regularizers.l2(0.5 * (cfg['weight_decay'])),
#             name=name,
#             dtype=input.dtype.base_dtype,
#         )
#         return layer(input)


def prediction_layer_stage(cfg, input, name, num_outputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        padding="SAME",
        activation_fn=None,
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(cfg["weight_decay"]),
    ):
        with tf.compat.v1.variable_scope(name):
            pred = slim.conv2d(
                input,
                num_outputs,
                kernel_size=[3, 3],
                stride=1,
            )
            return pred
