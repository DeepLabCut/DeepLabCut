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
"""PyTorch implementation of DeepLabCut.

``deeplabcut.api`` is the canonical entry point for the public API; the re-exports
below are kept so existing ``dlc_torch.<name>`` code keeps working while callers
migrate.

Attributes are attached lazily: eagerly re-exporting ``apis``, ``data`` and
``runners`` here made *any* import below this package (including the
pydantic-only ``config`` subpackage) pull in torch, torchvision, timm and
albumentations.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "apis",
        "benchmark",
        "config",
        "data",
        "metrics",
        "models",
        "modelzoo",
        "post_processing",
        "registry",
        "runners",
        "task",
        "utils",
    ],
    submod_attrs={
        "apis": [
            "VideoIterator",
            "analyze_image_folder",
            "analyze_images",
            "analyze_videos",
            "build_predictions_dataframe",
            "convert_detections2tracklets",
            "create_labeled_images",
            "create_tracking_dataset",
            "evaluate",
            "evaluate_network",
            "extract_maps",
            "extract_save_all_maps",
            "get_detector_inference_runner",
            "get_pose_inference_runner",
            "predict",
            "superanimal_analyze_images",
            "train",
            "train_network",
            "video_inference",
            "visualize_predictions",
        ],
        "config": [
            "GenSamplingConfig",
            "available_detectors",
            "available_models",
            "is_model_cond_top_down",
            "is_model_top_down",
        ],
        "data": [
            "COCOLoader",
            "COLLATE_FUNCTIONS",
            "DLCLoader",
            "GenerativeSampler",
            "Loader",
            "PoseDataset",
            "PoseDatasetParameters",
            "Snapshot",
            "build_transforms",
            "list_snapshots",
        ],
        "runners": [
            "DetectorInferenceRunner",
            "DetectorTrainingRunner",
            "DynamicCropper",
            "InferenceRunner",
            "PoseInferenceRunner",
            "PoseTrainingRunner",
            "TopDownDynamicCropper",
            "TorchSnapshotManager",
            "TrainingRunner",
            "build_inference_runner",
            "build_training_runner",
            "get_load_weights_only",
            "set_load_weights_only",
        ],
        "task": ["Task"],
        "utils": ["fix_seeds"],
    },
)
