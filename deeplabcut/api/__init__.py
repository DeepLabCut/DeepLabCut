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

from importlib import import_module
from typing import Any

__all__ = [
    "analyze_images",
    "analyze_time_lapse_frames",
    "analyze_videos",
    "convert_detections2tracklets",
    "create_pretrained_project",
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
    "analyzeskeleton",
    "filterpredictions",
    "extract_outlier_frames",
    "find_outliers_in_raw_data",
    "merge_datasets",
    "stitch_tracklets",
    "video_inference_superanimal",
]

_API_MODULES = {
    "analyze_images": "pose_estimation",
    "analyze_time_lapse_frames": "pose_estimation",
    "analyze_videos": "pose_estimation",
    "convert_detections2tracklets": "pose_estimation",
    "create_pretrained_project": "create_project",
    "create_tracking_dataset": "pose_estimation",
    "evaluate_network": "pose_estimation",
    "export_model": "pose_estimation",
    "extract_maps": "pose_estimation",
    "extract_save_all_maps": "pose_estimation",
    "return_evaluate_network_data": "pose_estimation",
    "return_train_network_path": "pose_estimation",
    "train_network": "pose_estimation",
    "visualize_locrefs": "pose_estimation",
    "visualize_paf": "pose_estimation",
    "visualize_scoremaps": "pose_estimation",
    "analyzeskeleton": "post_processing",
    "filterpredictions": "post_processing",
    "extract_outlier_frames": "refine_training",
    "find_outliers_in_raw_data": "refine_training",
    "merge_datasets": "refine_training",
    "stitch_tracklets": "refine_training",
    "video_inference_superanimal": "modelzoo_inference",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _API_MODULES.get(name, "pose_estimation")
    module = import_module(f".api.{module_name}", package="deeplabcut")
    return getattr(module, name)
