import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

try:
    import tf_keras.src.legacy_tf_layers as legacy_tf_layers
    sys.modules["tf_keras.legacy_tf_layers"] = legacy_tf_layers
except ImportError:
    # Older tf-keras didn’t use src/, so nothing to do
    pass