# Static and runtime export contract for deeplabcut's top-level API.
# lazy_loader.attach_stub reads this file at runtime.
# Keep public exports here rather than in a parallel Python mapping.

from .api.create_project import (
    create_pretrained_project as create_pretrained_project,
)
from .api.modelzoo_inference import (
    video_inference_superanimal as video_inference_superanimal,
)
from .api.pose_estimation import (
    analyze_images as analyze_images,
)
from .api.pose_estimation import (
    analyze_time_lapse_frames as analyze_time_lapse_frames,
)
from .api.pose_estimation import (
    analyze_videos as analyze_videos,
)
from .api.pose_estimation import (
    convert_detections2tracklets as convert_detections2tracklets,
)
from .api.pose_estimation import (
    create_tracking_dataset as create_tracking_dataset,
)
from .api.pose_estimation import (
    evaluate_network as evaluate_network,
)
from .api.pose_estimation import (
    export_model as export_model,
)
from .api.pose_estimation import (
    extract_maps as extract_maps,
)
from .api.pose_estimation import (
    extract_save_all_maps as extract_save_all_maps,
)
from .api.pose_estimation import (
    return_evaluate_network_data as return_evaluate_network_data,
)
from .api.pose_estimation import (
    return_train_network_path as return_train_network_path,
)
from .api.pose_estimation import (
    train_network as train_network,
)
from .api.pose_estimation import (
    visualize_locrefs as visualize_locrefs,
)
from .api.pose_estimation import (
    visualize_paf as visualize_paf,
)
from .api.pose_estimation import (
    visualize_scoremaps as visualize_scoremaps,
)
from .api.post_processing import analyzeskeleton as analyzeskeleton
from .api.post_processing import filterpredictions as filterpredictions
from .api.refine_training import (
    extract_outlier_frames as extract_outlier_frames,
)
from .api.refine_training import (
    find_outliers_in_raw_data as find_outliers_in_raw_data,
)
from .api.refine_training import merge_datasets as merge_datasets
from .api.refine_training import stitch_tracklets as stitch_tracklets
from .core.engine import Engine as Engine
from .create_project import add_new_videos as add_new_videos
from .create_project import create_new_project as create_new_project
from .create_project import create_new_project_3d as create_new_project_3d
from .create_project import (
    create_pretrained_human_project as create_pretrained_human_project,
)
from .create_project import load_demo_data as load_demo_data
from .generate_training_dataset.frame_extraction import (
    extract_frames as extract_frames,
)
from .generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
    create_multianimaltraining_dataset as create_multianimaltraining_dataset,
)
from .generate_training_dataset.trainingsetmanipulation import (
    adddatasetstovideolistandviceversa as adddatasetstovideolistandviceversa,
)
from .generate_training_dataset.trainingsetmanipulation import (
    check_labels as check_labels,
)
from .generate_training_dataset.trainingsetmanipulation import (
    comparevideolistsanddatafolders as comparevideolistsanddatafolders,
)
from .generate_training_dataset.trainingsetmanipulation import (
    create_training_dataset as create_training_dataset,
)
from .generate_training_dataset.trainingsetmanipulation import (
    create_training_dataset_from_existing_split as create_training_dataset_from_existing_split,
)
from .generate_training_dataset.trainingsetmanipulation import (
    create_training_model_comparison as create_training_model_comparison,
)
from .generate_training_dataset.trainingsetmanipulation import (
    dropannotationfileentriesduetodeletedimages as dropannotationfileentriesduetodeletedimages,
)
from .generate_training_dataset.trainingsetmanipulation import (
    dropduplicatesinannotatinfiles as dropduplicatesinannotatinfiles,
)
from .generate_training_dataset.trainingsetmanipulation import (
    dropimagesduetolackofannotation as dropimagesduetolackofannotation,
)
from .generate_training_dataset.trainingsetmanipulation import (
    dropunlabeledframes as dropunlabeledframes,
)
from .generate_training_dataset.trainingsetmanipulation import (
    mergeandsplit as mergeandsplit,
)
from .gui.launch_script import launch_dlc as launch_dlc
from .gui.tabs.label_frames import label_frames as label_frames
from .gui.tabs.label_frames import refine_labels as refine_labels
from .gui.tracklet_toolbox import refine_tracklets as refine_tracklets
from .gui.widgets import SkeletonBuilder as SkeletonBuilder
from .pose_estimation_3d.camera_calibration import (
    calibrate_cameras as calibrate_cameras,
)
from .pose_estimation_3d.camera_calibration import (
    check_undistortion as check_undistortion,
)
from .pose_estimation_3d.plotting3D import (
    create_labeled_video_3d as create_labeled_video_3d,
)
from .pose_estimation_3d.triangulation import triangulate as triangulate
from .pose_tracking_pytorch import transformer_reID as transformer_reID
from .utils import auxfun_videos as auxfun_videos
from .utils import auxiliaryfunctions as auxiliaryfunctions
from .utils.auxfun_multianimal import convert2_maDLC as convert2_maDLC
from .utils.auxfun_videos import CropVideo as CropVideo
from .utils.auxfun_videos import DownSampleVideo as DownSampleVideo
from .utils.auxfun_videos import ShortenVideo as ShortenVideo
from .utils.auxfun_videos import check_video_integrity as check_video_integrity
from .utils.auxfun_videos import collect_video_paths as collect_video_paths
from .utils.conversioncode import (
    analyze_videos_converth5_to_csv as analyze_videos_converth5_to_csv,
)
from .utils.conversioncode import (
    analyze_videos_converth5_to_nwb as analyze_videos_converth5_to_nwb,
)
from .utils.conversioncode import convertcsv2h5 as convertcsv2h5
from .utils.make_labeled_video import (
    create_labeled_video as create_labeled_video,
)
from .utils.make_labeled_video import (
    create_video_with_all_detections as create_video_with_all_detections,
)
from .utils.plotting import plot_trajectories as plot_trajectories
from .version import VERSION as VERSION
from .version import __version__ as __version__

# DEBUG has no canonical declaration in another module, so declare it directly.
DEBUG: bool
