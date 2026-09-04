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
"""Public API for DeepLabCut.

This is a partial migration. More public functions will be moved into
``deeplabcut.api`` in future releases. Functions not yet listed here
are still available at the top-level ``deeplabcut`` module as before.
"""

from __future__ import annotations

from deeplabcut.api.create_project import create_pretrained_project
from deeplabcut.api.modelzoo_inference import video_inference_superanimal
from deeplabcut.api.pose_estimation import (
    analyze_images,
    analyze_time_lapse_frames,
    analyze_videos,
    convert_detections2tracklets,
    create_tracking_dataset,
    evaluate_network,
    export_model,
    extract_maps,
    extract_save_all_maps,
    return_evaluate_network_data,
    return_train_network_path,
    train_network,
    visualize_locrefs,
    visualize_paf,
    visualize_scoremaps,
)
from deeplabcut.api.post_processing import analyzeskeleton, filterpredictions
from deeplabcut.api.refine_training import (
    extract_outlier_frames,
    find_outliers_in_raw_data,
    merge_datasets,
    stitch_tracklets,
)

__all__ = [
    "analyze_images",
    "analyze_time_lapse_frames",
    "analyze_videos",
    "analyzeskeleton",
    "convert_detections2tracklets",
    "create_pretrained_project",
    "create_tracking_dataset",
    "evaluate_network",
    "export_model",
    "extract_maps",
    "extract_outlier_frames",
    "extract_save_all_maps",
    "filterpredictions",
    "find_outliers_in_raw_data",
    "merge_datasets",
    "return_evaluate_network_data",
    "return_train_network_path",
    "stitch_tracklets",
    "train_network",
    "video_inference_superanimal",
    "visualize_locrefs",
    "visualize_paf",
    "visualize_scoremaps",
]
