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


import os

DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]
from deeplabcut.version import VERSION, __version__

print(f"Loading DLC {VERSION}...")

try:
    from deeplabcut.gui.launch_script import launch_dlc
    from deeplabcut.gui.tabs.label_frames import (
        label_frames,
        refine_labels,
    )
    from deeplabcut.gui.tracklet_toolbox import refine_tracklets
    from deeplabcut.gui.widgets import SkeletonBuilder
except (ModuleNotFoundError, ImportError):
    print("DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)")

from deeplabcut.core.engine import Engine
from deeplabcut.create_project import (
    add_new_videos,
    create_new_project,
    create_new_project_3d,
    create_pretrained_human_project,
    create_pretrained_project,
    load_demo_data,
)
from deeplabcut.generate_training_dataset import (
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
from deeplabcut.modelzoo.video_inference import video_inference_superanimal
from deeplabcut.utils import (
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

try:
    from deeplabcut.pose_tracking_pytorch import transformer_reID
except ModuleNotFoundError:
    import warnings

    warnings.warn(
        """
        As PyTorch is not installed, unsupervised identity learning will not be available.
        Please run `pip install torch`, or ignore this warning.
        """,
        stacklevel=2,
    )

# Train, evaluate & predict functions / all require TF
from deeplabcut.compat import (
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
from deeplabcut.pose_estimation_3d import (
    calibrate_cameras,
    check_undistortion,
    create_labeled_video_3d,
    triangulate,
)
from deeplabcut.post_processing import analyzeskeleton, filterpredictions
from deeplabcut.refine_training_dataset import (
    extract_outlier_frames,
    find_outliers_in_raw_data,
    merge_datasets,
)
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils.auxfun_videos import (
    CropVideo,
    DownSampleVideo,
    ShortenVideo,
    check_video_integrity,
)
