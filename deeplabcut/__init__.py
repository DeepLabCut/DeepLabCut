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

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any

logger = logging.getLogger(__name__)

# DEBUG="", "0", "false", "no" -> False
DEBUG = os.environ.get("DEBUG", "").strip().lower() not in {"", "0", "false", "no"}

from .version import VERSION, __version__

if DEBUG:
    logger.debug("Loading DLC %s", VERSION)

# -----------------------------------------------------------------------------
# Always-available public API
# -----------------------------------------------------------------------------

# Train / evaluate / predict functions (compat layer)
from .compat import (
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
from .core.engine import Engine
from .create_project import (
    add_new_videos,
    create_new_project,
    create_new_project_3d,
    create_pretrained_human_project,
    create_pretrained_project,
    load_demo_data,
)
from .generate_training_dataset import (
    adddatasetstovideolistandviceversa,
    check_labels,
    comparevideolistsanddatafolders,
    create_multianimaltraining_dataset,
    create_training_dataset,
    create_training_dataset_from_existing_split,
    create_training_model_comparison,
    dropannotationfileentriesduetodeletedimages,
    dropduplicatesinannotatinfiles,
    dropimagesduetolackofannotation,
    dropunlabeledframes,
    extract_frames,
    mergeandsplit,
)
from .modelzoo.video_inference import video_inference_superanimal
from .pose_estimation_3d import (
    calibrate_cameras,
    check_undistortion,
    create_labeled_video_3d,
    triangulate,
)
from .post_processing import analyzeskeleton, filterpredictions
from .refine_training_dataset import (
    extract_outlier_frames,
    find_outliers_in_raw_data,
    merge_datasets,
)
from .refine_training_dataset.stitch import stitch_tracklets
from .utils import (
    analyze_videos_converth5_to_csv,
    analyze_videos_converth5_to_nwb,
    auxfun_videos,
    auxiliaryfunctions,
    convert2_maDLC,
    convertcsv2h5,
    create_labeled_video,
    create_video_with_all_detections,
    plot_trajectories,
)
from .utils.auxfun_videos import (
    CropVideo,
    DownSampleVideo,
    ShortenVideo,
    check_video_integrity,
    collect_video_paths,
)

# -----------------------------------------------------------------------------
# Optional / lazy public API
# -----------------------------------------------------------------------------
# These names are part of the public API, but importing them may require
# optional GUI or torch dependencies, so we lazy load them.
#
# Example:
#   import deeplabcut as dlc
#   dlc.launch_dlc()           # imports GUI code lazily
#   dlc.transformer_reID(...)  # imports torch-dependent code lazily
# -----------------------------------------------------------------------------

_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    # GUI
    "launch_dlc": (".gui.launch_script", "launch_dlc"),
    "label_frames": (".gui.tabs.label_frames", "label_frames"),
    "refine_labels": (".gui.tabs.label_frames", "refine_labels"),
    "refine_tracklets": (".gui.tracklet_toolbox", "refine_tracklets"),
    "SkeletonBuilder": (".gui.widgets", "SkeletonBuilder"),
    # Optional torch feature
    "transformer_reID": (".pose_tracking_pytorch", "transformer_reID"),
}


def __getattr__(name: str) -> Any:
    """Lazily load optional public exports."""
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _OPTIONAL_EXPORTS[name]

    try:
        module = import_module(module_name, package=__name__)
        value = getattr(module, attr_name)
    except (ModuleNotFoundError, ImportError) as exc:
        if name in {
            "launch_dlc",
            "label_frames",
            "refine_labels",
            "refine_tracklets",
            "SkeletonBuilder",
        }:
            raise AttributeError(
                f"{name!r} is unavailable because DeepLabCut was loaded without GUI dependencies."
            ) from exc

        if name == "transformer_reID":
            raise AttributeError(
                f"{name!r} is unavailable because the PyTorch-based tracking dependencies are not installed."
            ) from exc

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    # Cache the resolved object so future access is fast
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Improve IDE / autocomplete discoverability."""
    return sorted(set(globals()) | set(__all__))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

_VERSION_EXPORTS = [
    "__version__",
    "VERSION",
    "DEBUG",
]

_CORE_EXPORTS = [
    "Engine",
]

_PROJECT_EXPORTS = [
    "add_new_videos",
    "create_new_project",
    "create_new_project_3d",
    "create_pretrained_human_project",
    "create_pretrained_project",
    "load_demo_data",
]

_DATASET_EXPORTS = [
    "adddatasetstovideolistandviceversa",
    "check_labels",
    "comparevideolistsanddatafolders",
    "create_multianimaltraining_dataset",
    "create_training_dataset",
    "create_training_dataset_from_existing_split",
    "create_training_model_comparison",
    "dropannotationfileentriesduetodeletedimages",
    "dropduplicatesinannotatinfiles",
    "dropimagesduetolackofannotation",
    "dropunlabeledframes",
    "extract_frames",
    "mergeandsplit",
]

_COMPAT_EXPORTS = [
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

_UTIL_EXPORTS = [
    "analyze_videos_converth5_to_csv",
    "analyze_videos_converth5_to_nwb",
    "auxfun_videos",
    "auxiliaryfunctions",
    "convert2_maDLC",
    "convertcsv2h5",
    "create_labeled_video",
    "create_video_with_all_detections",
    "plot_trajectories",
    "CropVideo",
    "DownSampleVideo",
    "ShortenVideo",
    "check_video_integrity",
]

_POST_PROCESSING_EXPORTS = [
    "analyzeskeleton",
    "filterpredictions",
    "extract_outlier_frames",
    "find_outliers_in_raw_data",
    "merge_datasets",
    "stitch_tracklets",
]

_THREE_D_EXPORTS = [
    "calibrate_cameras",
    "check_undistortion",
    "create_labeled_video_3d",
    "triangulate",
]

_MODELZOO_EXPORTS = [
    "video_inference_superanimal",
]

_OPTIONAL_API_EXPORTS = list(_OPTIONAL_EXPORTS)

__all__ = (
    _VERSION_EXPORTS
    + _CORE_EXPORTS
    + _PROJECT_EXPORTS
    + _DATASET_EXPORTS
    + _COMPAT_EXPORTS
    + _UTIL_EXPORTS
    + _POST_PROCESSING_EXPORTS
    + _THREE_D_EXPORTS
    + _MODELZOO_EXPORTS
    + _OPTIONAL_API_EXPORTS
)
