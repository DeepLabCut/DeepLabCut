import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "1")

try:
    import h5py
    # Optional diagnostic (can be removed later)
    print(f"✅ h5py preloaded successfully: {h5py.__version__}")
except Exception as e:
    # Continue gracefully if h5py isn't installed yet
    print(f"⚠️ Warning: failed to preload h5py: {e}")

try:
    import tf_keras.src.legacy_tf_layers as legacy_tf_layers
    sys.modules["tf_keras.legacy_tf_layers"] = legacy_tf_layers
except ImportError:
    # Older tf-keras didn’t use src/, so nothing to do
    pass