"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os

# Suppress tensorflow warning messages
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]
from deeplabcut.version import __version__, VERSION

print(f"Loading DLC {VERSION}...")

try:
    from deeplabcut.gui.tracklet_toolbox import refine_tracklets
    from deeplabcut.gui.launch_script import launch_dlc
    from deeplabcut.gui.tabs.label_frames import (
        label_frames,
        refine_labels,
    )
    from deeplabcut.gui.widgets import SkeletonBuilder
except (ModuleNotFoundError, ImportError):
    print(
        "DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)"
    )

from deeplabcut.create_project import (
    create_new_project,
    create_new_project_3d,
    add_new_videos,
    load_demo_data,
    create_pretrained_project,
    create_pretrained_human_project,
)
from deeplabcut.generate_training_dataset import (
    check_labels,
    create_training_dataset,
    extract_frames,
    mergeandsplit,
)
from deeplabcut.generate_training_dataset import (
    create_training_model_comparison,
    create_multianimaltraining_dataset,
)
from deeplabcut.generate_training_dataset import (
    dropannotationfileentriesduetodeletedimages,
    comparevideolistsanddatafolders,
    dropimagesduetolackofannotation,
    adddatasetstovideolistandviceversa,
    dropduplicatesinannotatinfiles,
    dropunlabeledframes
)
from deeplabcut.utils import (
    create_labeled_video,
    create_video_with_all_detections,
    plot_trajectories,
    auxiliaryfunctions,
    convert2_maDLC,
    convertcsv2h5,
    analyze_videos_converth5_to_csv,
    analyze_videos_converth5_to_nwb,
    auxfun_videos,
)

try:
    from deeplabcut.pose_tracking_pytorch import transformer_reID
except ModuleNotFoundError as e:
    import warnings
    warnings.warn(
        """
        As PyTorch is not installed, unsupervised identity learning will not be available.
        Please run `pip install torch`, or ignore this warning.
        """
    )

from deeplabcut.utils.auxfun_videos import (
    ShortenVideo,
    DownSampleVideo,
    CropVideo,
    check_video_integrity,
)

# Train, evaluate & predict functions / all require TF
from deeplabcut.pose_estimation_tensorflow import (
    train_network,
    return_train_network_path,
    evaluate_network,
    return_evaluate_network_data,
    analyze_videos,
    create_tracking_dataset,
    analyze_time_lapse_frames,
    convert_detections2tracklets,
    extract_maps,
    visualize_scoremaps,
    visualize_locrefs,
    visualize_paf,
    extract_save_all_maps,
    export_model,
)


from deeplabcut.pose_estimation_3d import (
    calibrate_cameras,
    check_undistortion,
    triangulate,
    create_labeled_video_3d,
)

from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.refine_training_dataset import (
    extract_outlier_frames,
    merge_datasets,
    find_outliers_in_raw_data,
)
from deeplabcut.post_processing import filterpredictions, analyzeskeleton

