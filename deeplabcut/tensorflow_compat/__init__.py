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
Compatibility layer for the legacy TensorFlow API. Remove this package when TF support is dropped.
"""

from deeplabcut.tensorflow_compat.pose_estimation import (
    analyze_images,
    analyze_time_lapse_frames,
    analyze_videos,
    convert_detections2tracklets,
    create_tracking_dataset,
    evaluate_network,
    export_model,
    extract_maps,
    extract_save_all_maps,
    get_available_aug_methods,
    return_evaluate_network_data,
    return_train_network_path,
    train_network,
    visualize_locrefs,
    visualize_paf,
    visualize_scoremaps,
)

__all__ = [
    "MakeInference_yaml",
    "MakeTest_pose_yaml",
    "MakeTrain_pose_yaml",
    "ParseYaml",
    "analyze_images",
    "analyze_time_lapse_frames",
    "analyze_videos",
    "convert_detections2tracklets",
    "create_multianimaltraining_dataset",
    "create_tracking_dataset",
    "create_training_dataset",
    "create_training_dataset_from_existing_split",
    "create_training_model_comparison",
    "evaluate_network",
    "export_model",
    "extract_maps",
    "extract_save_all_maps",
    "get_available_aug_methods",
    "return_evaluate_network_data",
    "return_train_network_path",
    "train_network",
    "visualize_locrefs",
    "visualize_paf",
    "visualize_scoremaps",
]
