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
from __future__ import annotations

import copy
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
from tqdm import tqdm

import deeplabcut.pose_estimation_pytorch.apis.utils as utils
import deeplabcut.pose_estimation_pytorch.runners.shelving as shelving
from deeplabcut.pose_estimation_pytorch.apis.ctd import (
    get_condition_provider,
    get_conditions_provider_for_video,
)
from deeplabcut.pose_estimation_pytorch.apis.tracklets import (
    convert_detections2tracklets,
)
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromModel
from deeplabcut.pose_estimation_pytorch.runners import (
    CTDTrackingConfig,
    DynamicCropper,
    InferenceRunner,
    TopDownDynamicCropper,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils import auxiliaryfunctions, VideoReader


class VideoIterator(VideoReader):
    """A class to iterate over videos, with possible added context"""

    def __init__(
        self,
        video_path: str | Path,
        context: list[dict[str, Any]] | None = None,
        cropping: list[int] | None = None,
    ) -> None:
        super().__init__(str(video_path))
        self._context = context
        self._index = 0
        self._crop = cropping is not None
        if self._crop:
            self.set_bbox(*cropping)

    def set_crop(self, cropping: list[int] | None = None) -> None:
        """Sets the cropping parameters for the video."""
        self._crop = cropping is not None
        if self._crop:
            self.set_bbox(*cropping)
        else:
            self.set_bbox(0, 1, 0, 1, relative=True)

    def get_context(self) -> list[dict[str, Any]] | None:
        if self._context is None:
            return None

        return copy.deepcopy(self._context)

    def set_context(self, context: list[dict[str, Any]] | None) -> None:
        if context is None:
            self._context = None
            return

        self._context = copy.deepcopy(context)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray | tuple[str, dict[str, Any]]:
        frame = self.read_frame(crop=self._crop)
        if frame is None:
            self._index = 0
            self.reset()
            raise StopIteration

        # Otherwise ValueError: At least one stride in the given numpy array is negative,
        # and tensors with negative strides are not currently supported. (You can probably
        # work around this by making a copy of your array  with array.copy().)
        frame = frame.copy()
        if self._context is None:
            self._index += 1
            return frame

        context = copy.deepcopy(self._context[self._index])
        self._index += 1
        return frame, context


def video_inference(
    video: str | Path | VideoIterator,
    pose_runner: InferenceRunner,
    detector_runner: InferenceRunner | None = None,
    cropping: list[int] | None = None,
    shelf_writer: shelving.ShelfWriter | None = None,
    robust_nframes: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Runs inference on a video

    Args:
        video: The video to analyze
        pose_runner: The pose runner to run inference with
        detector_runner: When the pose model is a top-down model, a detector runner can
            be given to obtain bounding boxes for the video. If the pose model is a
            top-down model and no detector_runner is given, the bounding boxes must
            already be set in the VideoIterator (see examples).
        cropping: Optionally, video inference can be run on a cropped version of the
            video. To do so, pass a list containing 4 elements to specify which area
            of the video should be analyzed: ``[xmin, xmax, ymin, ymax]``.
        shelf_writer: By default, data are dumped in a pickle file at the end of the
            video analysis. Passing a shelf manager writes data to disk on-the-fly
            using a "shelf" (a pickle-based, persistent, database-like object by
            default, resulting in constant memory footprint). The returned list is
            then empty.
        robust_nframes: Evaluate a video's number of frames in a robust manner. This
            option is slower (as the whole video is read frame-by-frame), but does not
            rely on metadata, hence its robustness against file corruption.

    Returns:
        Predictions for each frame in the video. If a shelf_manager is given, this list
        will be empty and the predictions will exclusively be stored in the file written
        by the shelf.

    Examples:
        Bottom-up video analysis:
        >>> import deeplabcut.pose_estimation_pytorch as pep
        >>> from deeplabcut.core.config import read_config_as_dict
        >>> model_cfg = read_config_as_dict("pytorch_config.yaml")
        >>> runner = pep.get_pose_inference_runner(model_cfg, "snapshot.pt")
        >>> video_predictions = pep.video_inference("video.mp4", runner)
        >>>

        Top-down video analysis:
        >>> import deeplabcut.pose_estimation_pytorch as pep
        >>> from deeplabcut.core.config import read_config_as_dict
        >>> model_cfg = read_config_as_dict("pytorch_config.yaml")
        >>> runner = pep.get_pose_inference_runner(model_cfg, "snapshot.pt")
        >>> d_runner = pep.get_pose_inference_runner(model_cfg, "snapshot-detector.pt")
        >>> video_predictions = pep.video_inference("video.mp4", runner, d_runner)
        >>>

        Top-Down pose estimation with pre-computed bounding boxes:
        >>> import numpy as np
        >>> import deeplabcut.pose_estimation_pytorch as pep
        >>> from deeplabcut.core.config import read_config_as_dict
        >>>
        >>> video_iterator = pep.VideoIterator("video.mp4")
        >>> video_iterator.set_context([
        >>>     { # frame 1 context
        >>>         "bboxes": np.array([[12, 17, 4, 5]]),  # format (x0, y0, w, h)
        >>>     },
        >>>     { # frame 1 context
        >>>         "bboxes": np.array([[12, 17, 4, 5], [18, 92, 54, 32]]),
        >>>     },
        >>>     ...
        >>> ])
        >>> model_cfg = read_config_as_dict("pytorch_config.yaml")
        >>> runner = pep.get_pose_inference_runner(model_cfg, "snapshot.pt")
        >>> video_predictions = pep.video_inference(video_iterator, runner)
        >>>
    """
    if not isinstance(video, VideoIterator):
        video = VideoIterator(str(video), cropping=cropping)
    elif cropping is not None:
        video.set_crop(cropping)

    n_frames = video.get_n_frames(robust=robust_nframes)
    vid_w, vid_h = video.dimensions
    print(f"Starting to analyze {video.video_path}")
    print(
        f"Video metadata: \n"
        f"  Overall # of frames:    {n_frames}\n"
        f"  Duration of video [s]:  {n_frames / max(1, video.fps):.2f}\n"
        f"  fps:                    {video.fps}\n"
        f"  resolution:             w={vid_w}, h={vid_h}\n"
    )

    if detector_runner is not None:
        print(f"Running detector with batch size {detector_runner.batch_size}")
        
        detector_progress = tqdm(video, desc="Detector")
        bbox_predictions = []
        for i, frame in enumerate(detector_progress):
            result = detector_runner.inference(images=[frame])
            bbox_predictions.extend(result)
        
        # PATCH: Ensure bbox_predictions is always length n_frames
        if len(bbox_predictions) < n_frames:
            print(f"[PATCH] Detector returned {len(bbox_predictions)} predictions for {n_frames} frames. Padding with empty bboxes.")
            for _ in range(n_frames - len(bbox_predictions)):
                bbox_predictions.append({'bboxes': np.zeros((0, 4))})
        elif len(bbox_predictions) > n_frames:
            print(f"[PATCH] Detector returned more predictions than frames. Truncating to {n_frames}.")
            bbox_predictions = bbox_predictions[:n_frames]
        video.set_context(bbox_predictions)

    print(f"Running pose prediction with batch size {pose_runner.batch_size}")
    if shelf_writer is not None:
        shelf_writer.open()
    
    pose_progress = tqdm(video, desc="Pose")
    predictions = []
    for i, frame in enumerate(pose_progress):
        result = pose_runner.inference(images=[frame])
        predictions.extend(result)
    
    if shelf_writer is not None:
        shelf_writer.close()

    if shelf_writer is None and len(predictions) != n_frames:
        frames_with_detections = sum(
            1 for pred in predictions if (
                ('bodyparts' in pred and pred['bodyparts'].shape[0] > 0) or
                ('bboxes' in pred and len(pred['bboxes']) > 0)
            )
        )
        logging.warning(
            f"Only {frames_with_detections} of {n_frames} frames had detections!"
        )

    return predictions


def analyze_videos(
    config: str,
    videos: str | list[str],
    videotype: str | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    save_as_csv: bool = False,
    in_random_order: bool = False,
    snapshot_index: int | str | None = None,
    detector_snapshot_index: int | str | None = None,
    device: str | None = None,
    destfolder: str | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    ctd_conditions: dict | CondFromModel | None = None,
    ctd_tracking: bool | dict | CTDTrackingConfig = False,
    top_down_dynamic: dict | None = None,
    modelprefix: str = "",
    use_shelve: bool = False,
    robust_nframes: bool = False,
    transform: A.Compose | None = None,
    auto_track: bool | None = True,
    n_tracks: int | None = None,
    animal_names: list[str] | None = None,
    calibrate: bool = False,
    identity_only: bool | None = False,
    overwrite: bool = False,
    cropping: list[int] | None = None,
    save_as_df: bool = False,
) -> str:
    """Makes prediction based on a trained network.

    The index of the trained network is specified by parameters in the config file
    (in particular the variable 'snapshot_index').

    Args:
        config: full path of the config.yaml file for the project
        videos: a str (or list of strings) containing the full paths to videos for
            analysis or a path to the directory, where all the videos with same
            extension are stored.
        videotype: checks for the extension of the video in case the input to the video
            is a directory. Only videos with this extension are analyzed. If left
            unspecified, keeps videos with extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv').
        shuffle: An integer specifying the shuffle index of the training dataset used for
            training the network.
        trainingsetindex: Integer specifying which TrainingsetFraction to use.
        save_as_csv: For multi-animal projects and when `auto_track=True`, passed
            along to the `stitch_tracklets` method to save tracks as CSV.
        in_random_order: Whether or not to analyze videos in a random order. This is
            only relevant when specifying a video directory in `videos`.
        device: the device to use for video analysis
        destfolder: specifies the destination folder for analysis data. If ``None``,
            the path of the video is used. Note that for subsequent analysis this
            folder also needs to be passed
        snapshot_index: index (starting at 0) of the snapshot to use to analyze the
            videos. To evaluate the last one, use -1. For example if we have
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
                - snapshot-best.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshot index is loaded from the project configuration.
        detector_snapshot_index: (only for top-down models) index of the detector
            snapshot to use, used in the same way as ``snapshot_index``
        dynamic: (state, detection threshold, margin) triplet. If the state is true,
            then dynamic cropping will be performed. That means that if an object is
            detected (i.e. any body part > detection threshold), then object boundaries
            are computed according to the smallest/largest x position and
            smallest/largest y position of all body parts. This  window is expanded by
            the margin and from then on only the posture within this crop is analyzed
            (until the object is lost, i.e. < detection threshold). The current position
            is utilized for updating the crop window for the next frame (this is why the
            margin is important and should be set large enough given the movement of the
            animal).
        ctd_conditions: Only for CTD models. If None, the configuration for the
            condition provider will be loaded from the pytorch_config file (under the
            "data": "conditions"). If the ctd_conditions is given as a dict, creates a
            CondFromModel from the dict. Otherwise, a CondFromModel can be given
            directly. Example configuration:
                ```
                ctd_conditions = {"shuffle": 17, "snapshot": "snapshot-best-190.pt"}
                ```
        ctd_tracking: Only for CTD models. Conditional top-down models can be used
            to directly track individuals. Poses from frame T are given as conditions
            for frame T+1. This also means a BU model is only needed to "initialize" the
            pose in the first frame, and for the remaining frames only the CTD model is
            needed. To configure conditional pose tracking differently, you can pass a
            CTDTrackingConfig instance.
        top_down_dynamic: Configuration for a top-down dynamic cropper. If None,
            top-down dynamic cropping is not used. Can only be used when running
            inference on a single animal. If an empty dict is given, default parameters
            are used. This is not recommended, as parameters should be customized for
            your data. Possible parameters are:
                "top_down_crop_size": tuple[int, int]
                    The (width, height) to resize the crop to. If not specified, will
                    be loaded from the `pytorch_cfg.yaml` for your top-down model. If
                    your model is not a top-down model, must be given.
                "patch_counts": tuple[int, int] (default: (3, 2))
                    The number of patches along the (width, height) of the images when
                    no crop is found.
                "patch_overlap": int (default: 50)
                    The amount of overlapping pixels between adjacent patches.
                "min_bbox_size": tuple[int, int] (default: (50, 50))
                    The minimum (width, height) for a detected bounding box.
                "threshold": float (default: 0.6)
                    The threshold score for bodyparts above which an individual is
                    considered to be detected.
                "margin": int (default: 25)
                    The margin to add around keypoints when generating bounding boxes.
                "min_hq_keypoints": int (default: 2)
                    The minimum number of keypoints above the threshold required for the
                    individual to be considered detected and a bbox to be computed.
                "bbox_from_hq": bool (default: False)
                    If True, only keypoints above the score threshold will be used to
                    compute the bounding boxes.
        modelprefix: directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        batch_size: the batch size to use for inference. Takes the value from the
            project config as a default.
        detector_batch_size: the batch size to use for detector inference. Takes the
            value from the project config as a default.
        transform: Optional custom transforms to apply to the video
        overwrite: Overwrite any existing videos
        use_shelve: By default, data are dumped in a pickle file at the end of the video
            analysis. Otherwise, data are written to disk on the fly using a "shelf";
            i.e., a pickle-based, persistent, database-like object by default, resulting
            in constant memory footprint.
        robust_nframes: Evaluate a video's number of frames in a robust manner. This
            option is slower (as the whole video is read frame-by-frame), but does not
            rely on metadata, hence its robustness against file corruption.
        auto_track: By default, tracking and stitching are automatically performed,
            producing the final h5 data file. This is equivalent to the behavior for
            single-animal projects.

            If ``False``, one must run ``convert_detections2tracklets`` and
            ``stitch_tracklets`` afterwards, in order to obtain the h5 file.
        n_tracks: Number of tracks to reconstruct. By default, taken as the number of
            individuals defined in the config.yaml. Another number can be passed if the
            number of animals in the video is different from the number of animals the
            model was trained on.
        animal_names: If you want the names given to individuals in the labeled data
            file, you can specify those names as a list here. If given and `n_tracks`
            is None, `n_tracks` will be set to `len(animal_names)`. If `n_tracks` is not
            None, then it must be equal to `len(animal_names)`. If it is not given, then
            `animal_names` will be loaded from the `individuals` in the project
            `config.yaml` file.
        identity_only: sub-call for auto_track. If ``True`` and animal identity was
            learned by the model, assembly and tracking rely exclusively on identity
            prediction.
        cropping: List of cropping coordinates as [x1, x2, y1, y2]. Note that the same
            cropping parameters will then be used for all videos. If different video
            crops are desired, run ``analyze_videos`` on individual videos with the
            corresponding cropping coordinates.
        save_as_df: Cannot be used when `use_shelve` is True. Saves the video
            predictions (before tracking results) to an H5 file containing a pandas
            DataFrame. If ``save_as_csv==True`` than the full predictions will also be
            saved in a CSV file.

    Returns:
        The scorer used to analyze the videos
    """
    # Create the output folder
    _validate_destfolder(destfolder)

    # Load the project configuration
    loader = DLCLoader(
        config,
        trainset_index=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )

    train_fraction = loader.project_cfg["TrainingFraction"][trainingsetindex]
    pose_cfg_path = loader.model_folder.parent / "test" / "pose_cfg.yaml"
    pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_path)

    snapshot_index, detector_snapshot_index = utils.parse_snapshot_index_for_analysis(
        loader.project_cfg, loader.model_cfg, snapshot_index, detector_snapshot_index,
    )

    if cropping is None and loader.project_cfg.get("cropping", False):
        cropping = (
            loader.project_cfg["x1"],
            loader.project_cfg["x2"],
            loader.project_cfg["y1"],
            loader.project_cfg["y2"],
        )

    # Get general project parameters
    multi_animal = loader.project_cfg["multianimalproject"]
    bodyparts = loader.model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = loader.model_cfg["metadata"]["unique_bodyparts"]
    individuals = loader.model_cfg["metadata"]["individuals"]
    max_num_animals = len(individuals)

    if device is not None:
        loader.model_cfg["device"] = device

    if batch_size is None:
        batch_size = loader.project_cfg.get("batch_size", 1)

    if not multi_animal:
        save_as_df = True
        if use_shelve:
            print(
                "The ``use_shelve`` parameter cannot be used for single animal "
                "projects. Setting ``use_shelve=False``."
            )
            use_shelve = False

    dynamic = DynamicCropper.build(*dynamic)
    if loader.pose_task != Task.BOTTOM_UP and dynamic is not None:
        print(
            "Turning off dynamic cropping. It should only be used for bottom-up "
            "pose estimation models, but you are using a top-down model. For top-down "
            "models, use the TopDownDynamicCropper with the `top_down_dynamic` arg."
        )
        dynamic = None

    if top_down_dynamic is not None:
        if loader.pose_task == Task.TOP_DOWN:
            td_cfg = loader.model_cfg["data"]["inference"].get(
                "top_down_crop",
                {"width": 256, "height": 256},
            )
            top_down_dynamic["top_down_crop_size"] = td_cfg["width"], td_cfg["height"]

        print(f"Creating a TopDownDynamicCropper with configuration {top_down_dynamic}")
        dynamic = TopDownDynamicCropper(**top_down_dynamic)

    try:
        snapshot = utils.get_model_snapshots(
            snapshot_index, loader.model_folder, loader.pose_task
        )[0]
    except (ValueError, IndexError) as e:
        print(f"Error loading snapshot with index {snapshot_index}: {e}")
        print("Attempting to find available snapshots...")
        
        # Try to get all available snapshots
        try:
            all_snapshots = utils.get_model_snapshots("all", loader.model_folder, loader.pose_task)
            if all_snapshots:
                # Try to find a "best" snapshot first
                best_snapshots = [s for s in all_snapshots if s.best]
                if best_snapshots:
                    snapshot = best_snapshots[0]
                    print(f"Found and using best snapshot: {snapshot.path}")
                else:
                    # Use the last available snapshot
                    snapshot = all_snapshots[-1]
                    print(f"No best snapshot found, using last available: {snapshot.path}")
            else:
                raise FileNotFoundError(f"No snapshots found in {loader.model_folder}")
        except Exception as fallback_error:
            raise FileNotFoundError(f"Failed to load any snapshots from {loader.model_folder}. Original error: {e}. Fallback error: {fallback_error}")

    # Additional validation for best snapshots
    if "best" in str(snapshot.path) and not snapshot.path.exists():
        print(f"Warning: Best snapshot path {snapshot.path} does not exist. Checking for alternative snapshots...")
        # Try to find any available snapshot
        try:
            all_snapshots = utils.get_model_snapshots("all", loader.model_folder, loader.pose_task)
            if all_snapshots:
                # Try to find a different best snapshot
                best_snapshots = [s for s in all_snapshots if s.best and s.path.exists()]
                if best_snapshots:
                    snapshot = best_snapshots[0]
                    print(f"Using alternative best snapshot: {snapshot.path}")
                else:
                    # Use the last available snapshot
                    snapshot = all_snapshots[-1]
                    print(f"Using alternative snapshot: {snapshot.path}")
            else:
                raise FileNotFoundError(f"No snapshots found in {loader.model_folder}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to find alternative snapshots: {e}")

    # Verify the snapshot file exists
    if not snapshot.path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot.path}")
    
    print(f"Successfully loaded snapshot: {snapshot.path}")

    # Load the BU model for the conditions provider
    cond_provider = None
    if loader.pose_task == Task.COND_TOP_DOWN:
        if ctd_conditions is None:
            cond_provider = get_condition_provider(
                condition_cfg=loader.model_cfg["data"]["conditions"],
                config=config,
            )
        elif isinstance(ctd_conditions, dict):
            cond_provider = get_condition_provider(
                condition_cfg=ctd_conditions, config=config,
            )
        else:
            cond_provider = ctd_conditions

    if isinstance(ctd_tracking, dict):
        # FIXME(niels) - add video FPS setting
        ctd_tracking = CTDTrackingConfig.build(ctd_tracking)

    print(f"Analyzing videos with {snapshot.path}")
    pose_runner = utils.get_pose_inference_runner(
        model_config=loader.model_cfg,
        snapshot_path=snapshot.path,
        max_individuals=max_num_animals,
        batch_size=batch_size,
        transform=transform,
        dynamic=dynamic,
        cond_provider=cond_provider,
        ctd_tracking=ctd_tracking,
    )

    detector_runner = None
    detector_path, detector_snapshot = None, None
    if loader.pose_task == Task.TOP_DOWN and dynamic is None:
        if detector_snapshot_index is None:
            raise ValueError(
                "Cannot run videos analysis for top-down models without a detector "
                "snapshot! Please specify your desired detector_snapshotindex in your "
                "project's configuration file."
            )

        if detector_batch_size is None:
            detector_batch_size = loader.project_cfg.get("detector_batch_size", 1)

        try:
            detector_snapshot = utils.get_model_snapshots(
                detector_snapshot_index, loader.model_folder, Task.DETECT
            )[0]
        except (ValueError, IndexError) as e:
            print(f"Error loading detector snapshot with index {detector_snapshot_index}: {e}")
            print("Attempting to find available detector snapshots...")
            
            # Try to get all available detector snapshots
            try:
                all_detector_snapshots = utils.get_model_snapshots("all", loader.model_folder, Task.DETECT)
                if all_detector_snapshots:
                    # Try to find a "best" detector snapshot first
                    best_detector_snapshots = [s for s in all_detector_snapshots if s.best]
                    if best_detector_snapshots:
                        detector_snapshot = best_detector_snapshots[0]
                        print(f"Found and using best detector snapshot: {detector_snapshot.path}")
                    else:
                        # Use the last available detector snapshot
                        detector_snapshot = all_detector_snapshots[-1]
                        print(f"No best detector snapshot found, using last available: {detector_snapshot.path}")
                else:
                    raise FileNotFoundError(f"No detector snapshots found in {loader.model_folder}")
            except Exception as fallback_error:
                raise FileNotFoundError(f"Failed to load any detector snapshots from {loader.model_folder}. Original error: {e}. Fallback error: {fallback_error}")

        # Additional validation for detector snapshots
        if "best" in str(detector_snapshot.path) and not detector_snapshot.path.exists():
            print(f"Warning: Best detector snapshot path {detector_snapshot.path} does not exist. Checking for alternative detector snapshots...")
            try:
                all_detector_snapshots = utils.get_model_snapshots("all", loader.model_folder, Task.DETECT)
                if all_detector_snapshots:
                    # Try to find a different best detector snapshot
                    best_detector_snapshots = [s for s in all_detector_snapshots if s.best and s.path.exists()]
                    if best_detector_snapshots:
                        detector_snapshot = best_detector_snapshots[0]
                        print(f"Using alternative best detector snapshot: {detector_snapshot.path}")
                    else:
                        # Use the last available detector snapshot
                        detector_snapshot = all_detector_snapshots[-1]
                        print(f"Using alternative detector snapshot: {detector_snapshot.path}")
                else:
                    raise FileNotFoundError(f"No detector snapshots found in {loader.model_folder}")
            except Exception as e:
                raise FileNotFoundError(f"Failed to find alternative detector snapshots: {e}")

        # Verify the detector snapshot file exists
        if not detector_snapshot.path.exists():
            raise FileNotFoundError(f"Detector snapshot file not found: {detector_snapshot.path}")

        print(f"  -> Using detector {detector_snapshot.path}")
        detector_runner = utils.get_detector_inference_runner(
            model_config=loader.model_cfg,
            snapshot_path=detector_snapshot.path,
            max_individuals=max_num_animals,
            batch_size=detector_batch_size,
        )

    dlc_scorer = loader.scorer(snapshot, detector_snapshot)
    print(f"Using scorer: {dlc_scorer}")

    # Reading video and init variables
    videos = utils.list_videos_in_folder(videos, videotype, shuffle=in_random_order)
    h5_files_created = False  # Track if any .h5 files were created
    
    for video in videos:
        if destfolder is None:
            output_path = video.parent
        else:
            output_path = Path(destfolder)

        output_prefix = video.stem + dlc_scorer
        output_pkl = output_path / f"{output_prefix}_full.pickle"

        video_iterator = VideoIterator(video, cropping=cropping)

        # Check if BU model pose predictions exist so the model does not need to be run
        if loader.pose_task == Task.COND_TOP_DOWN:
            vid_cond_provider = get_conditions_provider_for_video(cond_provider, video)
            if vid_cond_provider is not None:
                video_cond = vid_cond_provider.load_conditions()
                video_iterator.set_context([dict(cond_kpts=c) for c in video_cond])

        shelf_writer = None
        if use_shelve:
            shelf_writer = shelving.ShelfWriter(
                pose_cfg=pose_cfg,
                filepath=output_pkl,
                num_frames=video_iterator.get_n_frames(robust=robust_nframes),
            )

        if not overwrite and output_pkl.exists():
            print(f"Video {video} already analyzed at {output_pkl}!")
        else:
            runtime = [time.time()]
            predictions = video_inference(
                video=video_iterator,
                pose_runner=pose_runner,
                detector_runner=detector_runner,
                shelf_writer=shelf_writer,
                robust_nframes=robust_nframes,
            )
            runtime.append(time.time())
            metadata = _generate_metadata(
                cfg=loader.project_cfg,
                pytorch_config=loader.model_cfg,
                dlc_scorer=dlc_scorer,
                train_fraction=train_fraction,
                batch_size=batch_size,
                cropping=cropping,
                runtime=(runtime[0], runtime[1]),
                video=video_iterator,
                robust_nframes=robust_nframes,
            )

            with open(output_path / f"{output_prefix}_meta.pickle", "wb") as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

            if use_shelve and save_as_df:
                print("Can't ``save_as_df`` as ``use_shelve=True``. Skipping.")

            if not use_shelve:
                output_data = _generate_output_data(pose_cfg, predictions)
                with open(output_pkl, "wb") as f:
                    pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)

                if save_as_df:
                    create_df_from_prediction(
                        predictions=predictions,
                        multi_animal=multi_animal,
                        model_cfg=loader.model_cfg,
                        dlc_scorer=dlc_scorer,
                        output_path=output_path,
                        output_prefix=output_prefix,
                        save_as_csv=save_as_csv,
                    )
                    h5_files_created = True  # .h5 file was created

            if multi_animal:
                assemblies_path = output_path / f"{output_prefix}_assemblies.pickle"
                _generate_assemblies_file(
                    full_data_path=output_pkl,
                    output_path=assemblies_path,
                    num_bodyparts=len(bodyparts),
                    num_unique_bodyparts=len(unique_bodyparts),
                )

                # when running CTD tracking, don't auto-track as CTD did the tracking
                # for us!
                if ctd_tracking:
                    full_data = auxiliaryfunctions.read_pickle(output_pkl)
                    full_data_meta = full_data.pop("metadata")

                    num_frames = full_data_meta["nframes"]
                    str_width = full_data_meta["key_str_width"]

                    ctd_predictions = []
                    for i in range(num_frames):
                        frame_data = full_data.get("frame" + str(i).zfill(str_width))
                        if frame_data is None:
                            pose = np.full((len(individuals), len(bodyparts), 3), np.nan)
                            ctd_predictions.append(dict(bodyparts=pose))
                            continue

                        # there can't be unique bodyparts for CTD models
                        #   -> so coords has shape (num_bodyparts, num_idv, _)
                        coords = np.stack(frame_data["coordinates"][0], axis=0)
                        scores = np.stack(frame_data["confidence"], axis=0)
                        pose = np.concatenate([coords, scores], axis=-1)

                        # transpose to (num_idv, num_bodyparts, _)
                        pose = pose.transpose((1, 0, 2))

                        # add poses to the predictions
                        ctd_predictions.append(dict(bodyparts=pose))

                    create_df_from_prediction(
                        predictions=predictions,
                        multi_animal=multi_animal,
                        model_cfg=loader.model_cfg,
                        dlc_scorer=dlc_scorer,
                        output_path=output_path,
                        output_prefix=output_prefix + "_ctd",
                        save_as_csv=save_as_csv,
                    )
                    h5_files_created = True  # .h5 file was created for CTD tracking

                elif auto_track:
                    convert_detections2tracklets(
                        config=config,
                        videos=str(video),
                        videotype=videotype,
                        shuffle=shuffle,
                        trainingsetindex=trainingsetindex,
                        overwrite=False,
                        identity_only=identity_only,
                        destfolder=str(output_path),
                    )
                    stitch_tracklets(
                        config,
                        [str(video)],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        n_tracks=n_tracks,
                        animal_names=animal_names,
                        destfolder=str(output_path),
                        save_as_csv=save_as_csv,
                    )
                    h5_files_created = True  # .h5 file was created by stitch_tracklets

    if h5_files_created:
        print(
            "The videos are analyzed. Now your research can truly start!\n"
            "You can create labeled videos with 'create_labeled_video'.\n"
            "If the tracking is not satisfactory for some videos, consider expanding the "
            "training set. You can use the function 'extract_outlier_frames' to extract a "
            "few representative outlier frames.\n"
        )
    else:
        print(
            "No .h5 files were created during video analysis. Please check your code and "
            "ensure that the video inference and output generation are correct.\n"
        )

    return dlc_scorer


def create_df_from_prediction(
    predictions: list[dict[str, np.ndarray]],
    dlc_scorer: str,
    multi_animal: bool,
    model_cfg: dict,
    output_path: str | Path,
    output_prefix: str | Path,
    save_as_csv: bool = False,
) -> pd.DataFrame:
    # Check if any predictions were made
    if not predictions:
        raise ValueError(
            "No objects were detected in the video. This can happen if:\n"
            "1. The video doesn't contain the type of objects the model was trained to detect\n"
            "2. The objects are too small, blurry, or occluded\n"
            "3. The detector confidence threshold is too high\n"
            "4. The video quality is poor\n\n"
            "Try:\n"
            "- Using a different video with clearer objects\n"
            "- Adjusting the detector confidence threshold\n"
            "- Checking if the model is appropriate for your use case"
        )
    
    # Check if any predictions contain valid detections (non-empty bboxes)
    valid_predictions = []
    for pred in predictions:
        if "bboxes" in pred and len(pred["bboxes"]) > 0:
            valid_predictions.append(pred)
        elif "bodyparts" in pred and pred["bodyparts"].shape[0] > 0:
            valid_predictions.append(pred)
    
    if not valid_predictions:
        raise ValueError(
            "No objects were detected in the video. This can happen if:\n"
            "1. The video doesn't contain the type of objects the model was trained to detect\n"
            "2. The objects are too small, blurry, or occluded\n"
            "3. The detector confidence threshold is too high\n"
            "4. The video quality is poor\n\n"
            "Try:\n"
            "- Using a different video with clearer objects\n"
            "- Adjusting the detector confidence threshold\n"
            "- Checking if the model is appropriate for your use case"
        )
    
    # Ensure all predictions have the same shape by padding with zeros if needed
    max_individuals = max(p["bodyparts"].shape[0] for p in predictions) if predictions else 0
    num_bodyparts = predictions[0]["bodyparts"].shape[1] if predictions else 0
    
    # Pad all predictions to have the same number of individuals
    padded_predictions = []
    for p in predictions:
        current_individuals = p["bodyparts"].shape[0]
        if current_individuals < max_individuals:
            # Pad with zeros for missing individuals
            padding = np.zeros((max_individuals - current_individuals, num_bodyparts, 3))
            padded_bodyparts = np.concatenate([p["bodyparts"][..., :3], padding], axis=0)
        else:
            padded_bodyparts = p["bodyparts"][..., :3]
        padded_predictions.append(padded_bodyparts)
    
    pred_bodyparts = np.stack(padded_predictions)
    
    pred_unique_bodyparts = None
    if len(predictions) > 0 and "unique_bodyparts" in predictions[0]:
        pred_unique_bodyparts = np.stack([p["unique_bodyparts"] for p in predictions])

    output_h5 = Path(output_path) / f"{output_prefix}.h5"
    output_pkl = Path(output_path) / f"{output_prefix}_full.pickle"

    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    individuals = model_cfg["metadata"]["individuals"]
    n_individuals = len(individuals)

    print(f"Saving results in {output_h5} and {output_pkl}")
    coords = ["x", "y", "likelihood"]
    cols = [[dlc_scorer], bodyparts, coords]
    cols_names = ["scorer", "bodyparts", "coords"]

    if multi_animal:
        cols.insert(1, individuals)
        cols_names.insert(1, "individuals")

    results_df_index = pd.MultiIndex.from_product(cols, names=cols_names)
    pred_bodyparts = pred_bodyparts[:, :n_individuals]
    df = pd.DataFrame(
        pred_bodyparts.reshape((len(pred_bodyparts), -1)),
        columns=results_df_index,
        index=range(len(pred_bodyparts)),
    )
    if pred_unique_bodyparts is not None:
        unique_columns = [dlc_scorer], ["single"], unique_bodyparts, coords
        df_u = pd.DataFrame(
            pred_unique_bodyparts.reshape((len(pred_unique_bodyparts), -1)),
            columns=pd.MultiIndex.from_product(unique_columns, names=cols_names),
            index=range(len(pred_unique_bodyparts)),
        )
        df = df.join(df_u, how="outer")

    df.to_hdf(output_h5, key="df_with_missing", format="table", mode="w")
    if save_as_csv:
        df.to_csv(output_h5.with_suffix(".csv"))
    return df


def _generate_assemblies_file(
    full_data_path: Path,
    output_path: Path,
    num_bodyparts: int,
    num_unique_bodyparts: int,
) -> None:
    """Generates the assemblies file from predictions"""
    if full_data_path.exists():
        with open(full_data_path, "rb") as f:
            data = pickle.load(f)

    else:
        data = shelving.ShelfReader(full_data_path)
        data.open()

    num_frames = data["metadata"]["nframes"]
    str_width = data["metadata"].get("key_str_width")
    if str_width is None:
        keys = [k for k in data.keys() if k != "metadata"]
        str_width = len(keys[0]) - len("frame")

    assemblies = dict(single=dict())
    for frame_index in range(num_frames):
        frame_key = "frame" + str(frame_index).zfill(str_width)
        predictions = data[frame_key]

        keypoint_preds = predictions["coordinates"][0]
        keypoint_scores = predictions["confidence"]

        bpts = np.stack(keypoint_preds[:num_bodyparts])
        scores = np.stack(keypoint_scores[:num_bodyparts])
        preds = np.concatenate([bpts, scores], axis=-1)

        keypoint_id_scores = predictions.get("identity")
        if keypoint_id_scores is not None:
            keypoint_id_scores = np.stack(keypoint_id_scores[:num_bodyparts])
            keypoint_pred_ids = np.argmax(keypoint_id_scores, axis=2)
            keypoint_pred_ids = np.expand_dims(keypoint_pred_ids, axis=-1)
        else:
            num_bpts, num_preds = preds.shape[:2]
            keypoint_pred_ids = -np.ones((num_bpts, num_preds, 1))

        # reshape to (num_preds, num_bpts, 4)
        preds = np.concatenate([preds, keypoint_pred_ids], axis=-1)
        preds = preds.transpose((1, 0, 2))

        # remove all-missing predictions
        mask = ~np.all(preds < 0, axis=(1, 2))
        preds = preds[mask]

        assemblies[frame_index] = preds

        if num_unique_bodyparts > 0:
            unique_bpts = np.stack(keypoint_preds[num_bodyparts:])
            unique_scores = np.stack(keypoint_scores[num_bodyparts:])
            unique_preds = np.concatenate([unique_bpts, unique_scores], axis=-1)
            unique_preds = unique_preds.transpose((1, 0, 2))
            assemblies["single"][frame_index] = unique_preds[0]  # single prediction

    with open(output_path, "wb") as file:
        pickle.dump(assemblies, file, pickle.HIGHEST_PROTOCOL)

    if isinstance(data, shelving.ShelfReader):
        data.close()


def _validate_destfolder(destfolder: str | None) -> None:
    """Checks that the destfolder for video analysis is valid"""
    if destfolder is not None and destfolder != "":
        output_folder = Path(destfolder)
        if not output_folder.exists():
            print(f"Creating the output folder {output_folder}")
            output_folder.mkdir(parents=True)

        assert Path(
            output_folder
        ).is_dir(), f"Output folder must be a directory: you passed '{output_folder}'"


def _generate_metadata(
    cfg: dict,
    pytorch_config: dict,
    dlc_scorer: str,
    train_fraction: int,
    batch_size: int,
    cropping: list[int] | None,
    runtime: tuple[float, float],
    video: VideoIterator,
    robust_nframes: bool = False,
) -> dict:
    w, h = video.dimensions
    if cropping is None:
        cropping_parameters = [0, w, 0, h]
    else:
        if not len(cropping) == 4:
            raise ValueError(
                "The cropping parameters should be exactly 4 values: [x_min, x_max, "
                f"y_min, y_max]. Found {cropping}"
            )
        cropping_parameters = cropping

    metadata = {
        "start": runtime[0],
        "stop": runtime[1],
        "run_duration": runtime[1] - runtime[0],
        "Scorer": dlc_scorer,
        "pytorch-config": pytorch_config,
        "fps": video.fps,
        "batch_size": batch_size,
        "frame_dimensions": (w, h),
        "nframes": video.get_n_frames(robust=robust_nframes),
        "iteration (active-learning)": cfg["iteration"],
        "training set fraction": train_fraction,
        "cropping": cropping is not None,
        "cropping_parameters": cropping_parameters,
        "individuals": pytorch_config["metadata"]["individuals"],
        "bodyparts": pytorch_config["metadata"]["bodyparts"],
        "unique_bodyparts": pytorch_config["metadata"]["unique_bodyparts"],
    }
    return {"data": metadata}


def _generate_output_data(
    pose_config: dict,
    predictions: list[dict[str, np.ndarray]],
) -> dict:
    str_width = int(np.ceil(np.log10(len(predictions))))
    output = {
        "metadata": {
            "nms radius": pose_config.get("nmsradius"),
            "minimal confidence": pose_config.get("minconfidence"),
            "sigma": pose_config.get("sigma", 1),
            "PAFgraph": pose_config.get("partaffinityfield_graph"),
            "PAFinds": pose_config.get(
                "paf_best",
                np.arange(len(pose_config.get("partaffinityfield_graph", []))),
            ),
            "all_joints": [[i] for i in range(len(pose_config["all_joints"]))],
            "all_joints_names": [
                pose_config["all_joints_names"][i]
                for i in range(len(pose_config["all_joints"]))
            ],
            "nframes": len(predictions),
            "key_str_width": str_width,
        }
    }

    for frame_num, frame_predictions in enumerate(predictions):
        key = "frame" + str(frame_num).zfill(str_width)
        # shape (num_assemblies, num_bpts, 3)
        bodyparts = frame_predictions["bodyparts"]
        # shape (num_bpts, num_assemblies, 3)
        bodyparts = bodyparts.transpose((1, 0, 2))
        coordinates = [bpt[:, :2] for bpt in bodyparts]
        scores = [bpt[:, 2:3] for bpt in bodyparts]

        # full pickle has bodyparts and unique bodyparts in same array
        num_unique = 0
        if "unique_bodyparts" in frame_predictions:
            unique_bpts = frame_predictions["unique_bodyparts"].transpose((1, 0, 2))
            coordinates += [bpt[:, :2] for bpt in unique_bpts]
            scores += [bpt[:, 2:] for bpt in unique_bpts]
            num_unique = len(unique_bpts)

        output[key] = {
            "coordinates": (coordinates,),
            "confidence": scores,
            "costs": None,
        }

        if "bboxes" in frame_predictions:
            output[key]["bboxes"] = frame_predictions["bboxes"]
        if "bbox_scores" in frame_predictions:
            output[key]["bbox_scores"] = frame_predictions["bbox_scores"]

        if "identity_scores" in frame_predictions:
            # Reshape id scores from (num_assemblies, num_bpts, num_individuals)
            # to the original DLC full pickle format: (num_bpts, num_assem, num_ind)
            id_scores = frame_predictions["identity_scores"]
            id_scores = id_scores.transpose((1, 0, 2))
            output[key]["identity"] = [bpt_id_scores for bpt_id_scores in id_scores]

            if num_unique > 0:
                # needed for create_video_with_all_detections to display unique bpts
                num_assem, num_ind = id_scores.shape[1:]
                output[key]["identity"] += [
                    -1 * np.ones((num_assem, num_ind)) for i in range(num_unique)
                ]

    return output
