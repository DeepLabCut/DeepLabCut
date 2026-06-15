#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

"""Smoke test for TensorFlow when optional TF extras are installed."""

import pytest

tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow not installed (use a project extra such as .[tf])",
)


def test_tensorflow_imports_and_has_matmul() -> None:
    assert tf.__version__
    a = tf.constant([[1.0, 2.0]])
    b = tf.constant([[3.0], [4.0]])
    c = tf.matmul(a, b)

    if tf.executing_eagerly():
        result = c.numpy()
    else:
        with tf.compat.v1.Session() as sess:
            result = sess.run(c)

    assert (result == [[11.0]]).all()


def test_tf_slim_imports_and_has_conv2d() -> None:
    try:
        import tf_slim as slim
    except ImportError as e:
        raise AssertionError("tf_slim is not installed or not importable") from e

    assert slim.conv2d(tf.constant([[[[1.0]]]]), 1, kernel_size=[1, 1], stride=1).shape == (1, 1, 1, 1)


def test_tf_keras_imports_and_has_regularizers() -> None:
    try:
        import tf_keras as keras
    except ImportError as e:
        raise AssertionError("tf_keras is not installed or not importable") from e
    import numpy as np

    assert keras.regularizers.l2(0.01).l2 == np.array(0.01, dtype="float32")
