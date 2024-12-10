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

from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.apis.convert_detections_to_tracklets import (
    convert_detections2tracklets,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_model_snapshots,
    get_inference_runners,
    get_scorer_name,
    get_scorer_uid,
    list_videos_in_folder,
    parse_snapshot_index_for_analysis,
)
import deeplabcut.pose_estimation_pytorch.runners.shelving as shelving
from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils import auxiliaryfunctions, VideoReader


class VideoIterator(VideoReader):
    """A class to iterate over videos, with possible added context"""

    def __init__(
        self, video_path: str | Path, context: list[dict[str, Any]] | None = None, cropping: list[int] | None = None
    ) -> None:
        super().__init__(str(video_path))
        self._context = context
        self._index = 0
        self._crop = cropping is not None
        if self._crop:
            self.set_bbox(*cropping)

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
    task: Task,
    pose_runner: InferenceRunner,
    detector_runner: InferenceRunner | None = None,
    cropping: list[int] | None = None,
    shelf_writer: shelving.ShelfWriter | None = None,
    robust_nframes: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Runs inference on a video

    Args:
        video: The video to analyze
        task: The pose task to run (bottom-up or top-down)
        pose_runner: The pose runner to run inference with
        detector_runner: When ``task==Task.TOP_DOWN``, the detector runner to obtain
            bounding boxes for the video.
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
    """
    if not isinstance(video, VideoIterator):
        video = VideoIterator(str(video), cropping=cropping)

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

    if task == Task.TOP_DOWN:
        # Get bounding boxes for context
        if detector_runner is None:
            raise ValueError("Must use a detector for top-down video analysis")

        print(f"Running detector with batch size {detector_runner.batch_size}")
        bbox_predictions = detector_runner.inference(images=tqdm(video))
        video.set_context(bbox_predictions)

    print(f"Running pose prediction with batch size {pose_runner.batch_size}")
    if shelf_writer is not None:
        shelf_writer.open()

    predictions = pose_runner.inference(images=tqdm(video), shelf_writer=shelf_writer)
    if shelf_writer is not None:
        shelf_writer.close()

    if shelf_writer is None and len(predictions) != n_frames:
        tip_url = "https://deeplabcut.github.io/DeepLabCut/docs/recipes/io.html"
        header = "#tips-on-video-re-encoding-and-preprocessing"
        logging.warning(
            f"The video metadata indicates that there {n_frames} in the video, but "
            f"only {len(predictions)} were able to be processed. This can happen if "
            "the video is corrupted. You can try to fix the issue by re-encoding your "
            f"video (tips on how to do that: {tip_url}{header})"
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
    modelprefix: str = "",
    use_shelve: bool = False,
    robust_nframes: bool = False,
    transform: A.Compose | None = None,
    auto_track: bool | None = True,
    n_tracks: int | None = None,
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
        dynamic: TODO(niels)
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
        calibrate: TODO(niels)
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
    cfg = auxiliaryfunctions.read_config(config)
    project_path = Path(cfg["project_path"])
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    model_folder = project_path / auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        cfg,
        modelprefix=modelprefix,
        engine=Engine.PYTORCH,
    )
    train_folder = model_folder / "train"

    # Read the inference configuration, load the model
    model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
    model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
    pose_task = Task(model_cfg["method"])

    pose_cfg_path = model_folder / "test" / "pose_cfg.yaml"
    pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_path)

    snapshot_index, detector_snapshot_index = parse_snapshot_index_for_analysis(
        cfg, model_cfg, snapshot_index, detector_snapshot_index,
    )

    if cropping is None and cfg.get("cropping", False):
        cropping = cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]

    # Get general project parameters
    multi_animal = cfg["multianimalproject"]
    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    individuals = model_cfg["metadata"]["individuals"]
    with_identity = model_cfg["metadata"]["with_identity"]
    max_num_animals = len(individuals)

    if device is not None:
        model_cfg["device"] = device

    if batch_size is None:
        batch_size = cfg.get("batch_size", 1)

    if not multi_animal:
        save_as_df = True
        if use_shelve:
            print(
                "The ``use_shelve`` parameter cannot be used for single animal "
                "projects. Setting ``use_shelve=False``."
            )
            use_shelve = False

    snapshot = get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
    print(f"Analyzing videos with {snapshot.path}")
    detector_path, detector_snapshot = None, None
    if pose_task == Task.TOP_DOWN:
        if detector_snapshot_index is None:
            raise ValueError(
                "Cannot run videos analysis for top-down models without a detector "
                "snapshot! Please specify your desired detector_snapshotindex in your "
                "project's configuration file."
            )

        if detector_batch_size is None:
            detector_batch_size = cfg.get("detector_batch_size", 1)

        detector_snapshot = get_model_snapshots(
            detector_snapshot_index, train_folder, Task.DETECT
        )[0]
        detector_path = detector_snapshot.path
        print(f"  -> Using detector {detector_path}")

    dlc_scorer = get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot.path,
        max_individuals=max_num_animals,
        num_bodyparts=len(bodyparts),
        num_unique_bodyparts=len(unique_bodyparts),
        batch_size=batch_size,
        with_identity=with_identity,
        transform=transform,
        detector_batch_size=detector_batch_size,
        detector_path=detector_path,
        detector_transform=None,
    )

    # Reading video and init variables
    videos = list_videos_in_folder(videos, videotype, shuffle=in_random_order)
    for video in videos:
        if destfolder is None:
            output_path = video.parent
        else:
            output_path = Path(destfolder)

        output_prefix = video.stem + dlc_scorer
        output_pkl = output_path / f"{output_prefix}_full.pickle"

        video_iterator = VideoIterator(video)

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
                task=pose_task,
                detector_runner=detector_runner,
                cropping=cropping,
                shelf_writer=shelf_writer,
                robust_nframes=robust_nframes,
            )
            runtime.append(time.time())
            metadata = _generate_metadata(
                cfg=cfg,
                pytorch_config=model_cfg,
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
                        model_cfg=model_cfg,
                        dlc_scorer=dlc_scorer,
                        output_path=output_path,
                        output_prefix=output_prefix,
                        save_as_csv=save_as_csv,
                    )

            if multi_animal:
                _generate_assemblies_file(
                    full_data_path=output_pkl,
                    output_path=output_path / f"{output_prefix}_assemblies.pickle",
                    num_bodyparts=len(bodyparts),
                    num_unique_bodyparts=len(unique_bodyparts),
                )

                if auto_track:
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
                        destfolder=str(output_path),
                        save_as_csv=save_as_csv,
                    )

    print(
        "The videos are analyzed. Now your research can truly start!\n"
        "You can create labeled videos with 'create_labeled_video'.\n"
        "If the tracking is not satisfactory for some videos, consider expanding the "
        "training set. You can use the function 'extract_outlier_frames' to extract a "
        "few representative outlier frames.\n"
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
    pred_bodyparts = np.stack(
        [p["bodyparts"][..., :3] for p in predictions]
    )
    pred_unique_bodyparts = None
    if len(predictions) > 0 and "unique_bodyparts" in predictions[0]:
        pred_unique_bodyparts = np.stack(
            [p["unique_bodyparts"] for p in predictions]
        )

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
        unique_columns = [dlc_scorer], ['single'], unique_bodyparts, coords
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
