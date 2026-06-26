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
"""
Public API for DeepLabCut.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "analyze_images",
    "analyze_time_lapse_frames",
    "analyze_videos",
    "convert_detections2tracklets",
    "create_tracking_dataset",
    "evaluate_network",
    "export_model",
    "extract_maps",
    "extract_save_all_maps",
    "return_evaluate_network_data",
    "return_train_network_path",
    "train_network",
    "visualize_locrefs",
    "visualize_paf",
    "visualize_scoremaps",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from deeplabcut.api import pose_estimation

    return getattr(pose_estimation, name)
