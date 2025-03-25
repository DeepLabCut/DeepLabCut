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
"""Code to create tracking datasets for ReID model training"""
from pathlib import Path

from tqdm import tqdm

import deeplabcut.pose_estimation_pytorch.apis.utils as utils
import deeplabcut.pose_estimation_pytorch.data as data
import deeplabcut.pose_estimation_pytorch.data.postprocessor as postprocessing
import deeplabcut.pose_estimation_pytorch.models as models
import deeplabcut.pose_estimation_pytorch.runners as runners
import deeplabcut.pose_estimation_pytorch.runners.shelving as shelving
from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_tracking_pytorch import create_triplets_dataset


def build_feature_extraction_runner(
    loader: data.Loader,
    snapshot_path: str | Path,
    device: str,
    batch_size: int = 1,
) -> runners.PoseInferenceRunner:
    """Builds a runner to extract backbone features for poses of individuals

    Args:
        loader: The loader for the model to use.
        snapshot_path: The path of the snapshot to use.
        device: The device on which to run pose estimation.
        batch_size: The batch size to run pose estimation with.

    Returns:
        A PoseInferenceRunner that will return features for extracted pose.
    """
    num_features = loader.model_cfg["model"]["backbone_output_channels"]
    num_bodyparts = len(loader.model_cfg["metadata"]["bodyparts"])
    top_down = loader.pose_task != Task.BOTTOM_UP
    rescale_mode = postprocessing.RescaleAndOffset.Mode.KEYPOINT
    if top_down:
        rescale_mode = postprocessing.RescaleAndOffset.Mode.KEYPOINT_TD
        data_cfg = loader.model_cfg["data"]["inference"]
        crop_cfg = data_cfg.get("top_down_crop", {})
        width, height = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
        preprocessor = data.build_top_down_preprocessor(
            color_mode=loader.model_cfg["data"]["colormode"],
            transform=data.build_transforms(data_cfg),
            top_down_crop_size=(width, height),
            top_down_crop_margin=crop_cfg.get("margin", 0),
        )
    else:
        preprocessor = data.build_bottom_up_preprocessor(
            loader.model_cfg["data"]["colormode"],
            data.build_transforms(loader.model_cfg["data"]["inference"])
        )

    postprocessor = postprocessing.ComposePostprocessor(
        [
            postprocessing.PrepareBackboneFeatures(top_down=top_down),
            postprocessing.ConcatenateOutputs(
                keys_to_concatenate={
                    "bodyparts": ("bodypart", "poses"),
                    "features": ("backbone", "bodypart_features"),
                },
                empty_shapes={
                    "bodyparts": (num_bodyparts, 3),
                    "features": (num_bodyparts, num_features),
                },
                create_empty_outputs=True,
            ),
            postprocessing.RescaleAndOffset(["bodyparts"], rescale_mode),
        ]
    )

    runner = runners.build_inference_runner(
        task=loader.pose_task,
        model=models.PoseModel.build(loader.model_cfg["model"]),
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        load_weights_only=loader.model_cfg["runner"].get("load_weights_only", None),
    )
    assert isinstance(runner, runners.PoseInferenceRunner), (
        f"Failed to build inference runner: got type {type(runner)}"
    )

    # Set the model to output backbone features
    runner.model.output_features = True

    return runner


def extract_features_for_video(
    runner: runners.PoseInferenceRunner,
    video: VideoIterator,
    shelf_writer: shelving.FeatureShelfWriter,
    detector_runner: runners.DetectorInferenceRunner | None = None,
) -> None:
    """Extracts backbone features for predicted keypoints in a video.

    Args:
        video: The video for which to extract backbone features.
        runner: The inference runner with which to extract backbone features.
        shelf_writer: The ShelfWriter used to extract features.
        detector_runner: For top-down models, the detector to use to predict bboxes.
    """
    if detector_runner is not None:
        print(f"Running detector with batch size {detector_runner.batch_size}")
        bbox_predictions = detector_runner.inference(images=tqdm(video))
        video.set_context(bbox_predictions)

    shelf_writer.open()
    runner.inference(tqdm(video), shelf_writer=shelf_writer)
    shelf_writer.close()


def create_tracking_dataset(
    config: str,
    videos: list[str] | list[Path],
    track_method: str,
    videotype: str = "",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    destfolder: str | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    cropping: list[int] | None = None,
    modelprefix: str = "",
    robust_nframes: bool = False,
    n_triplets: int = 1000,
) -> str:
    """Creates a tracking dataset to train a ReID tracklet stitcher.

    Args:
        config: Full path of the config.yaml file for the project
        videos: A str (or list of strings) containing the full paths to videos from
            which to create the tracking dataset or a path to the directory, where all
            the videos with same extension are stored.
        track_method: Specifies the tracker used to generate the pose estimation data.
            Must be either 'box', 'skeleton', or 'ellipse'.
        videotype: Checks for the extension of the video in case the input to the video
            is a directory. Only videos with this extension are analyzed. If left
            unspecified, keeps videos with extensions ('avi', 'mp4', 'mov', 'mpeg',
            'mkv').
        shuffle: An integer specifying the shuffle index of the training dataset used
            for training the network.
        trainingsetindex: Integer specifying which TrainingsetFraction to use.
        destfolder: Specifies the destination folder for the tracking data. If ``None``,
            the path of the video is used. Note that for subsequent analysis this
            folder also needs to be passed.
        batch_size: The batch size to use for inference. Takes the value from the
            project config as a default.
        detector_batch_size: The batch size to use for detector inference. Takes the
            value from the project config as a default.
        cropping: List of cropping coordinates as [x1, x2, y1, y2]. Note that the same
            cropping parameters will then be used for all videos. If different video
            crops are desired, run ``analyze_videos`` on individual videos with the
            corresponding cropping coordinates.
        modelprefix: Directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        robust_nframes: Evaluate a video's number of frames in a robust manner. This
            option is slower (as the whole video is read frame-by-frame), but does not
            rely on metadata, hence its robustness against file corruption.
        n_triplets: The number of triplets to extract for the dataset.

    Returns:
        The scorer used to analyze the videos.
    """
    loader = data.DLCLoader(
        config,
        trainset_index=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )
    test_cfg_path = loader.model_folder.parent / "test" / "pose_cfg.yaml"
    test_cfg = read_config_as_dict(test_cfg_path)

    snapshot_index, detector_snapshot_index = utils.parse_snapshot_index_for_analysis(
        loader.project_cfg, loader.model_cfg, None, None,
    )
    snapshot = utils.get_model_snapshots(
        snapshot_index, loader.model_folder, loader.pose_task,
    )[0]

    if cropping is None and loader.project_cfg.get("cropping", False):
        cropping = (
            loader.project_cfg["x1"],
            loader.project_cfg["x2"],
            loader.project_cfg["y1"],
            loader.project_cfg["y2"],
        )

    output_folder = None
    if destfolder is not None and destfolder != "":
        output_folder = Path(destfolder)

    if batch_size is None:
        batch_size = loader.project_cfg["batch_size"]

    device = utils.resolve_device(loader.model_cfg)
    runner = build_feature_extraction_runner(
        loader, snapshot.path, device, batch_size=batch_size
    )

    detector_runner = None
    detector_snapshot = None
    if loader.pose_task == Task.TOP_DOWN:
        if detector_batch_size is None:
            detector_batch_size = loader.project_cfg.get("detector_batch_size", 1)

        detector_snapshot = utils.get_model_snapshots(
            detector_snapshot_index, loader.model_folder, Task.DETECT,
        )[0]
        detector_runner = utils.get_detector_inference_runner(
            model_config=loader.model_cfg,
            snapshot_path=detector_snapshot.path,
            batch_size=detector_batch_size,
            device=device,
        )

    dlc_scorer = utils.get_scorer_name(
        loader.project_cfg,
        shuffle,
        loader.train_fraction,
        snapshot_uid=utils.get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )

    videos = utils.list_videos_in_folder(videos, videotype)
    for video_path in videos:
        print(f"Loading {video_path}")
        video = VideoIterator(video_path, cropping=cropping)

        nx, ny = video.dimensions
        nframes = video.get_n_frames(robust=robust_nframes)
        duration = video.calc_duration(robust=robust_nframes)
        fps = video.fps
        if robust_nframes:
            fps = nframes / duration

        print(f"Duration of video [s]: {duration:.2f}, recorded with {fps:.2f} fps!")
        print(f"Overall # of frames: {nframes} found with (before cropping)")
        print(f"Frame dimensions: {nx} x {ny}")

        if output_folder is None:
            output_folder = Path(video.video_path).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        output_prefix = Path(video_path).stem + dlc_scorer
        output_filepath = output_folder / f"{output_prefix}_bpt_features.pickle"

        shelf_writer = shelving.FeatureShelfWriter(
            test_cfg,
            output_filepath,
            num_frames=video.get_n_frames(robust=robust_nframes),
        )
        extract_features_for_video(
            runner, video, shelf_writer, detector_runner=detector_runner
        )

    create_triplets_dataset(
        videos,
        dlc_scorer,
        track_method,
        n_triplets=n_triplets,
        destfolder=destfolder,
    )
    return dlc_scorer
