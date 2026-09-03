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
"""Internal helpers for DeepLabCut."""

# TODO @deruyter92 2026-08-31: We should consider removing these imports alltogether
# and stop advertising a public API for them. But this would be a breaking change.
# For now, star-everything imports are replaced by lazy loading.
# see https://github.com/DeepLabCut/DeepLabCut/pull/3459
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "auxfun_models",
        "auxfun_multianimal",
        "auxfun_videos",
        "auxiliaryfunctions",
        "auxiliaryfunctions_3d",
        "conversioncode",
        "frameselectiontools",
        "make_labeled_video",
        "multiprocessing",
        "pandas_future_mode",
        "plotting",
        "pseudo_label",
        "skeleton",
        "video_processor",
        "visualization",
    ],
)
