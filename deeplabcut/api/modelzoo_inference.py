#
# DeepLabCut Toolbox (deeplabcut.org)
# (c) A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Public API for DeepLabCut modelzoo functions."""

from __future__ import annotations

from deeplabcut.api._tf_routing import with_tensorflow_fallback
from deeplabcut.core.deprecation import renamed_parameter
from deeplabcut.modelzoo.video_inference import (
    video_inference_superanimal as _impl,
)


@with_tensorflow_fallback(
    when=lambda *a, **kw: kw.get("model_name") == "dlcrnet",
    tensorflow_module="deeplabcut.tensorflow_compat.superanimal_inference",
    tensorflow_name="video_inference_superanimal_tf",
    dropped_params=["scale_list"],
)
@renamed_parameter(old="videotype", new="video_extensions", since="3.0.0")
def video_inference_superanimal(*args, **kwargs):
    """Video inference using pretrained SuperAnimal models.

    Delegates to :func:`deeplabcut.modelzoo.video_inference.video_inference_superanimal`.
    """
    return _impl(*args, **kwargs)


video_inference_superanimal.__doc__ = _impl.__doc__
