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
import pickle
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd

from deeplabcut.pose_estimation_pytorch.apis.convert_detections_to_tracklets import (
    convert_detections2tracklets,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_detector_snapshots,
    get_model_snapshots,
    get_runners,
    list_videos_in_folder,
)
from deeplabcut.pose_estimation_pytorch.runners import Runner
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, VideoReader


class VideoIterator(VideoReader):
    """A class to iterate over videos, with possible added context"""

    def __init__(
        self, video_path: str, context: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(video_path)
        self._context = context
        self._index = 0

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
        frame = self.read_frame()
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
    video_path: str | Path,
    task: str,
    pose_runner: Runner,
    detector_runner: Runner | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Runs inference on a video"""
    video = VideoIterator(str(video_path))
    n_frames = video.get_n_frames()
    vid_w, vid_h = video.dimensions
    print(
        f"Video metadata: \n"
        f"  n_frames:   {n_frames}\n"
        f"  fps:        {video.fps}\n"
        f"  resolution: w={vid_w}, h={vid_h}\n"
    )

    if task == "TD":
        # Get bounding boxes for context
        if detector_runner is None:
            raise ValueError("Must use a detector for top-down video analysis")

        bbox_predictions = detector_runner.inference(images=video)
        video.set_context(bbox_predictions)

    predictions = pose_runner.inference(images=video)
    poses = np.stack([p["bodyparts"] for p in predictions])
    unique_poses = None
    if len(predictions) > 0 and "unique_bodyparts" in predictions[0]:
        unique_poses = np.stack([p["unique_bodyparts"] for p in predictions])
    return poses, unique_poses


def analyze_videos(
    config: str,
    videos: Union[str, List[str]],
    videotype: Optional[str] = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshotindex: Optional[int] = None,
    device: Optional[str] = None,
    destfolder: Optional[str] = None,
    batchsize: Optional[int] = None,
    modelprefix: str = "",
    transform: Optional[A.Compose] = None,
    auto_track: Optional[bool] = True,
    identity_only: Optional[bool] = False,
    overwrite: bool = False,
) -> List[Tuple[str, pd.DataFrame]]:
    """Makes prediction based on a trained network.

    # TODO:
        - allow batch size greater than 1
        - other options such as save_as_csv
        - pass detector path or detector runner
        - add TQDM to runner

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
        device: the device to use for video analysis
        destfolder: specifies the destination folder for analysis data. If ``None``,
            the path of the video is used. Note that for subsequent analysis this
            folder also needs to be passed
        snapshotindex: index (starting at 0) of the snapshot to use to analyze the
            videos. To evaluate the last one, use -1. For example if we have
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshotindex is loaded from the project configuration.
        modelprefix: directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        batchsize: the batch size to use for inference. Takes the value from the
            PyTorch config as a default
        transform: Optional custom transforms to apply to the video
        overwrite: Overwrite any existing videos
        auto_track: By default, tracking and stitching are automatically performed,
            producing the final h5 data file. This is equivalent to the behavior for
            single-animal projects.

            If ``False``, one must run ``convert_detections2tracklets`` and
            ``stitch_tracklets`` afterwards, in order to obtain the h5 file.
        identity_only: sub-call for auto_track. If ``True`` and animal identity was
            learned by the model, assembly and tracking rely exclusively on identity
            prediction.

    Returns:
        A list containing tuples (video_name, df_video_predictions)
    """
    # Create the output folder
    _create_output_folder(destfolder)

    # Load the project configuration
    cfg = auxiliaryfunctions.read_config(config)
    project_path = Path(cfg["project_path"])
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    model_folder = project_path / auxiliaryfunctions.get_model_folder(
        train_fraction, shuffle, cfg, modelprefix=modelprefix
    )
    model_path = _get_model_path(model_folder, snapshotindex, cfg)
    model_epochs = int(model_path.stem.split("-")[-1])
    dlc_scorer, dlc_scorer_legacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        trainingsiterations=model_epochs,
        modelprefix=modelprefix,
    )
    # Get general project parameters
    bodyparts = auxiliaryfunctions.get_bodyparts(cfg)
    unique_bodyparts = auxiliaryfunctions.get_unique_bodyparts(cfg)
    individuals = cfg.get("individuals", ["animal"])
    max_num_animals = len(individuals)
    num_keypoints = len(bodyparts)

    # Read the inference configuration, load the model
    pytorch_config = auxiliaryfunctions.read_plainconfig(
        model_folder / "train" / "pytorch_config.yaml"
    )
    pose_cfg_path = model_folder / "test" / "pose_cfg.yaml"
    pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_path)
    method = pytorch_config.get("method", "BU").upper()

    if device is not None:
        pytorch_config["device"] = device

    detector_path = None
    if method == "TD":
        # TODO: Choose which detector to use
        detector_path = _get_detector_path(model_folder, -1, cfg)

    print(f"Analyzing videos with {model_path}")
    pose_runner, detector_runner = get_runners(
        pytorch_config=pytorch_config,
        snapshot_path=str(model_path),
        with_unique_bodyparts=(len(unique_bodyparts) > 0),
        transform=transform,
        detector_path=detector_path,
        detector_transform=None,
    )

    # Reading video and init variables
    videos = list_videos_in_folder(videos, videotype)
    results = []
    for video in videos:
        if destfolder is None:
            output_path = video.parent
        else:
            output_path = Path(destfolder)

        output_prefix = video.stem + dlc_scorer
        output_h5 = output_path / f"{output_prefix}.h5"
        output_pkl = output_path / f"{output_prefix}_full.pickle"
        if not overwrite and output_pkl.exists():
            print(f"Video already analyzed at {output_pkl}!")
        else:
            runtime = [time.time()]
            predictions, unique_predictions = video_inference(
                video_path=video,
                pose_runner=pose_runner,
                task=method,
                detector_runner=detector_runner,
            )
            runtime.append(time.time())

            print(f"Inference is done for {video}! Saving results...")
            metadata = _generate_metadata(
                cfg=cfg,
                pytorch_config=pytorch_config,
                dlc_scorer=dlc_scorer,
                train_fraction=train_fraction,
                batch_size=batchsize,
                runtime=(runtime[0], runtime[1]),
                video=VideoReader(str(video)),
            )

            coordinate_labels = ["x", "y", "likelihood"]
            if len(individuals) > 1:
                print("Extracting ", len(individuals), "instances per bodypart")
                # first has empty suffix for backwards compatibility
                individual_suffixes = [str(s + 1) for s in range(len(individuals))]
                individual_suffixes[0] = ""
                coordinate_labels = [
                    coord_label + s
                    for s in individual_suffixes
                    for coord_label in coordinate_labels
                ]

            results_df_index = pd.MultiIndex.from_product(
                [
                    [dlc_scorer],
                    auxiliaryfunctions.get_bodyparts(cfg),
                    coordinate_labels,
                ],
                names=["scorer", "bodyparts", "coords"],
            )
            df = pd.DataFrame(
                predictions.reshape((len(predictions), -1)),
                columns=results_df_index,
                index=range(len(predictions)),
            )
            if unique_predictions is not None:
                coordinate_labels_unique = ["x", "y", "likelihood"]
                results_unique_df_index = pd.MultiIndex.from_product(
                    [
                        [dlc_scorer],
                        auxiliaryfunctions.get_unique_bodyparts(cfg),
                        coordinate_labels_unique,
                    ],
                    names=["scorer", "bodyparts", "coords"],
                )
                df_u = pd.DataFrame(
                    unique_predictions.reshape((len(unique_predictions), -1)),
                    columns=results_unique_df_index,
                    index=range(len(unique_predictions)),
                )
                df = df.join(df_u, how="outer")

            df.to_hdf(str(output_h5), "df_with_missing", format="table", mode="w")
            results.append((str(video), df))
            output_data = _generate_output_data(pose_cfg, predictions)
            _ = auxfun_multianimal.SaveFullMultiAnimalData(
                output_data, metadata, str(output_h5)
            )

            if cfg["multianimalproject"] and len(individuals) > 1:
                output_ass = output_path / f"{output_prefix}_assemblies.pickle"
                assemblies = {}
                for i, prediction in enumerate(predictions):
                    extra_column = np.full(
                        (prediction.shape[0], prediction.shape[1], 1),
                        -1.0,
                        dtype=np.float32,
                    )
                    ass = np.concatenate((prediction, extra_column), axis=-1)
                    assemblies[i] = ass

                if unique_predictions is not None:
                    assemblies["single"] = {}
                    for i, unique_prediction in enumerate(unique_predictions):
                        extra_column = np.full(
                            (unique_prediction.shape[1], 1), -1.0, dtype=np.float32
                        )
                        ass = np.concatenate(
                            (unique_prediction[0], extra_column), axis=-1
                        )
                        assemblies["single"][i] = ass

                with open(output_ass, "wb") as handle:
                    pickle.dump(assemblies, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if auto_track:
                    convert_detections2tracklets(
                        config=config,
                        videos=str(video),
                        videotype=videotype,
                        shuffle=shuffle,
                        trainingsetindex=trainingsetindex,
                        overwrite=False,
                        identity_only=identity_only,
                        destfolder=destfolder,
                    )
                    stitch_tracklets(
                        config,
                        [str(video)],
                        videotype,
                        shuffle,
                        trainingsetindex,
                        destfolder=destfolder,
                    )

            else:
                results_df_index = pd.MultiIndex.from_product(
                    [
                        [dlc_scorer],
                        pose_cfg["all_joints_names"],
                        ["x", "y", "likelihood"],
                    ],
                    names=["scorer", "bodyparts", "coords"],
                )
                df = pd.DataFrame(
                    np.array(predictions).reshape((len(predictions), -1)),
                    columns=results_df_index,
                    index=range(len(predictions)),
                )
                df.to_hdf(str(output_h5), "df_with_missing", format="table", mode="w")
                results.append((str(video), df))
    return results


def _create_output_folder(output_folder: Optional[Path]) -> None:
    if output_folder is not None:
        output_folder = Path(output_folder)
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
    runtime: Tuple[float, float],
    video: VideoReader,
) -> dict:
    w, h = video.dimensions
    cropping = cfg.get("cropping", False)
    if cropping:
        cropping_parameters = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]
    else:
        cropping_parameters = [0, w, 0, h]

    metadata = {
        "start": runtime[0],
        "stop": runtime[1],
        "run_duration": runtime[1] - runtime[0],
        "Scorer": dlc_scorer,
        "pytorch-config": pytorch_config,
        "fps": video.fps,
        "batch_size": batch_size,
        "frame_dimensions": (w, h),
        "nframes": video.get_n_frames(),
        "iteration (active-learning)": cfg["iteration"],
        "training set fraction": train_fraction,
        "cropping": cropping,
        "cropping_parameters": cropping_parameters,
    }
    return {"data": metadata}


def _get_model_path(model_folder: Path, snapshot_index: int, config: dict) -> Path:
    trained_models = get_model_snapshots(model_folder / "train")

    if snapshot_index is None:
        snapshot_index = config["snapshotindex"]

    if snapshot_index == "all":
        print(
            "snapshotindex is set to 'all' in the config.yaml file. Running video "
            "analysis with all snapshots is very costly! Use the function "
            "'evaluate_network' to choose the best the snapshot. For now, changing "
            "snapshot index to -1. To evaluate another snapshot, you can change the "
            "value in the config file or call `analyze_videos` with your desired "
            "snapshot index."
        )
        snapshot_index = -1

    assert isinstance(
        snapshot_index, int
    ), f"snapshotindex must be an integer but was '{snapshot_index}'"
    return trained_models[snapshot_index]


def _get_detector_path(
    model_folder: Path, snapshot_index: int | str, config: dict | None
) -> Path:
    trained_models = get_detector_snapshots(model_folder / "train")

    if snapshot_index is None:
        snapshot_index = config["snapshotindex"]

    if snapshot_index == "all":
        print(
            "snapshotindex is set to 'all' in the config.yaml file. Running video "
            "analysis with all snapshots is very costly! Use the function "
            "'evaluate_network' to choose the best the snapshot. For now, changing "
            "snapshot index to -1. To evaluate another snapshot, you can change the "
            "value in the config file or call `analyze_videos` with your desired "
            "snapshot index."
        )
        snapshot_index = -1

    assert isinstance(
        snapshot_index, int
    ), f"snapshotindex must be an integer but was '{snapshot_index}'"
    return trained_models[snapshot_index]


def _generate_output_data(pose_config: dict, predictions: np.ndarray) -> dict:
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
        }
    }

    str_width = int(np.ceil(np.log10(len(predictions))))
    for frame_num, frame_predictions in enumerate(predictions):
        key = "frame" + str(frame_num).zfill(str_width)
        output[key] = frame_predictions.squeeze()

        # TODO: Do we want to keep the same format as in the TensorFlow version?
        #  On the one hand, it's "more" backwards compatible.
        #  On the other, might as well simplify the code. These files should only be loaded
        #    by the PyTorch version, and only predictions made by PyTorch models should be
        #    loaded using them
        # p_bodypart_indv = np.transpose(frame_predictions.squeeze(), axes=[1, 0, 2])
        # coords = [
        #     bodypart_predictions[:, :2] for bodypart_predictions in p_bodypart_indv
        # ]
        # scores = [
        #     bodypart_predictions[:, 2:] for bodypart_predictions in p_bodypart_indv
        # ]
        # output[key] = {
        #     "coordinates": (coords,),
        #     "confidence": scores,
        #     "costs": None,
        # }

    return output
