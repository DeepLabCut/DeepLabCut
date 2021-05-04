import tensorflow as tf


def prediction_layer(cfg, input, name, num_outputs):
    with tf.compat.v1.variable_scope(name):
        layer = tf.keras.layers.Conv2DTranspose(
            filters=num_outputs,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (cfg['weight_decay'])),
            name=name,
            dtype=input.dtype.base_dtype,
        )
        return layer(input)


### New DLCNet Addition: multi-stage decoder
def prediction_layer_stage(cfg, input, name, num_outputs):
    with tf.compat.v1.variable_scope(name):
        layer = tf.keras.layers.Conv2D(
            filters=num_outputs,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.5 * (cfg['weight_decay'])),
            name=name,
            dtype=input.dtype.base_dtype,
        )
        return layer(input)
