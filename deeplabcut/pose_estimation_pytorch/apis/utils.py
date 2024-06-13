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

import logging
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import pandas as pd

from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.dlcloader import (
    build_dlc_dataframe_columns,
)
from deeplabcut.pose_estimation_pytorch.data.postprocessor import (
    build_bottom_up_postprocessor,
    build_detector_postprocessor,
    build_top_down_postprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
    build_bottom_up_preprocessor,
    build_top_down_preprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.runners import (
    build_inference_runner,
    InferenceRunner,
)
from deeplabcut.pose_estimation_pytorch.runners.snapshots import (
    Snapshot,
    TorchSnapshotManager,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.utils import resolve_device
from deeplabcut.utils import auxfun_videos, auxiliaryfunctions


def parse_snapshot_index_for_analysis(
    cfg: dict,
    model_cfg: dict,
    snapshot_index: int | str | None,
    detector_snapshot_index: int | str | None,
) -> tuple[int, int | None]:
    """Gets the index of the snapshots to use for data analysis (e.g. video analysis)

    Args:
        cfg: The project configuration.
        model_cfg: The model configuration.
        snapshot_index: The index of the snapshot to use, if one was given by the user.
        detector_snapshot_index: The index of the detector snapshot to use, if one
            was given by the user.

    Returns:
        snapshot_index: the snapshot index to use for analysis
        detector_snapshot_index: the detector index to use for analysis, or None if no
            detector should be used
    """
    if snapshot_index is None:
        snapshot_index = cfg["snapshotindex"]
    if snapshot_index == "all":
        logging.warning(
            "snapshotindex is set to 'all' (in the config.yaml file or as given to "
            "`analyze_...`). Running data analysis with all snapshots is very "
            "costly! Use the function 'evaluate_network' to choose the best the "
            "snapshot. For now, changing snapshot index to -1. To evaluate another "
            "snapshot, you can change the value in the config file or call "
            "`analyze_videos` or `analyze_images` with your desired snapshot index."
        )
        snapshot_index = -1

    pose_task = Task(model_cfg["method"])
    if pose_task == Task.TOP_DOWN:
        if detector_snapshot_index is None:
            detector_snapshot_index = cfg.get("detector_snapshotindex", -1)

        if detector_snapshot_index == "all":
            logging.warning(
                f"detector_snapshotindex is set to '{detector_snapshot_index}' (in the "
                "config.yaml file or as given to `analyze_...`). Running data analysis "
                "with all snapshots is very costly! Use 'evaluate_network' to choose "
                "the best detector snapshot. For now, changing the detector snapshot "
                "index to -1. To evaluate another detector snapshot, you can change "
                "the value in the config file or call `analyze_videos` or "
                "`analyze_images` with your desired detector snapshot index."
            )
            detector_snapshot_index = -1

    else:
        detector_snapshot_index = None

    return snapshot_index, detector_snapshot_index


def return_train_network_path(
    config: str, shuffle: int = 1, trainingsetindex: int = 0, modelprefix: str = ""
) -> tuple[Path, Path, Path]:
    """
    Args:
        config: Full path of the config.yaml file as a string.
        shuffle: The shuffle index to select for training
        trainingsetindex: Which TrainingsetFraction to use (note that TrainingFraction
            is a list in config.yaml)
        modelprefix: the modelprefix for the model

    Returns:
        the path to the training pytorch pose configuration file
        the path to the test pytorch pose configuration file
        the path to the folder containing the snapshots
    """
    cfg = auxiliaryfunctions.read_config(config)
    project_path = Path(cfg["project_path"])
    train_frac = cfg["TrainingFraction"][trainingsetindex]
    model_folder = auxiliaryfunctions.get_model_folder(
        train_frac, shuffle, cfg, engine=Engine.PYTORCH, modelprefix=modelprefix
    )
    return (
        project_path / model_folder / "train" / "pytorch_config.yaml",
        project_path / model_folder / "test" / "pose_cfg.yaml",
        project_path / model_folder / "train",
    )


def get_model_snapshots(
    index: int | str,
    model_folder: Path,
    task: Task,
) -> list[Snapshot]:
    """
    Args:
        index: Passing an index returns the snapshot with that index (where snapshots
            based on their number of training epochs, and the last snapshot is the
            "best" model based on validation metrics if one exists). Passing "best"
            returns the best snapshot from the training run. Passing "all" returns all
            snapshots.
        model_folder: The path to the folder containing the snapshots
        task: The task for which to return the snapshot

    Returns:
        If index=="all", returns all snapshots. Otherwise, returns a list containing a
        single snapshot, with the desired index.

    Raises:
        ValueError: If the index given is not valid
        ValueError: If index=="best" but there is no saved best model
    """
    snapshot_manager = TorchSnapshotManager(model_folder=model_folder, task=task)
    if isinstance(index, str) and index.lower() == "best":
        best_snapshot = snapshot_manager.best()
        if best_snapshot is None:
            raise ValueError(f"No best snapshot found in {model_folder}")
        snapshots = [best_snapshot]
    elif isinstance(index, str) and index.lower() == "all":
        snapshots = snapshot_manager.snapshots(include_best=True)
    elif isinstance(index, int):
        all_snapshots = snapshot_manager.snapshots(include_best=True)
        if (
            len(all_snapshots) == 0
            or len(all_snapshots) <= index
            or (index < 0 and len(all_snapshots) < -index)
        ):
            names = [s.path.name for s in all_snapshots]
            raise ValueError(
                f"Found {len(all_snapshots)} snapshots in {model_folder} (with names "
                f"{names}) with prefix {snapshot_manager.task.snapshot_prefix}. Could "
                f"not return snapshot with index {index}."
            )

        snapshots = [all_snapshots[index]]
    else:
        raise ValueError(f"Invalid snapshotindex: {index}")

    return snapshots


def get_scorer_uid(snapshot: Snapshot, detector_snapshot: Snapshot | None) -> str:
    """
    Args:
        snapshot: the snapshot for which to get the scorer UID
        detector_snapshot: if a top-down model is used with a detector, the detector
            snapshot for which to get the scorer UID

    Returns:
        the uid to use for the scorer
    """
    snapshot_id = f"snapshot_{snapshot.uid()}"
    if detector_snapshot is not None:
        detect_id = detector_snapshot.uid()
        snapshot_id = f"detector_{detect_id}_{snapshot_id}"
    return snapshot_id


def get_scorer_name(
    cfg: dict,
    shuffle: int,
    train_fraction: float,
    snapshot_index: int | None = None,
    detector_index: int | None = None,
    snapshot_uid: str | None = None,
    modelprefix: str = "",
) -> str:
    """Get the scorer name for a particular PyTorch DeepLabCut shuffle

    Args:
        cfg: The project configuration.
        shuffle: The index of the shuffle for which to get the scorer
        train_fraction: The training fraction for the shuffle.
        snapshot_index: The index of the snapshot used. If None, the value is loaded
            from the project's config.yaml file.
        detector_index: For top-down models, the index of the detector used. If None,
            the value is loaded from the project's config.yaml file.
        snapshot_uid: If the snapshot_uid is not None, this value will be used instead
            of loading the snapshot and detector with given indices and calling
            utils.get_scorer_uid.
        modelprefix: The model prefix, if one was used.

    Returns:
        the scorer name
    """
    model_dir = Path(cfg["project_path"]) / auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        cfg,
        engine=Engine.PYTORCH,
        modelprefix=modelprefix,
    )
    train_dir = model_dir / "train"
    model_cfg = read_config_as_dict(str(train_dir / Engine.PYTORCH.pose_cfg_name))
    net_type = model_cfg["net_type"]
    pose_task = Task(model_cfg["method"])

    if snapshot_uid is None:
        if snapshot_index is None:
            snapshot_index = auxiliaryfunctions.get_snapshot_index_for_scorer(
                "snapshotindex", cfg["snapshotindex"]
            )
        if detector_index is None:
            detector_index = auxiliaryfunctions.get_snapshot_index_for_scorer(
                "detector_snapshotindex", cfg["detector_snapshotindex"]
            )

        snapshot = get_model_snapshots(snapshot_index, train_dir, pose_task)[0]
        detector_snapshot = None
        if detector_index is not None and pose_task == Task.TOP_DOWN:
            detector_snapshot = get_model_snapshots(
                detector_index, train_dir, Task.DETECT
            )[0]

        snapshot_uid = get_scorer_uid(snapshot, detector_snapshot)

    task, date = cfg["Task"], cfg["date"]
    name = "".join([p.capitalize() for p in net_type.split("_")])
    return f"DLC_{name}_{task}{date}shuffle{shuffle}_{snapshot_uid}"


def list_videos_in_folder(
    data_path: str | list[str], video_type: str | None
) -> list[Path]:
    """
    TODO
    """
    if not isinstance(data_path, list):
        data_path = [data_path]
    video_paths = [Path(p) for p in data_path]

    videos = []
    for video_path in video_paths:
        if video_path.is_dir():
            if video_type is None:
                video_suffixes = ["." + ext for ext in auxfun_videos.SUPPORTED_VIDEOS]
            else:
                video_suffixes = [video_type]

            video_suffixes = [
                s if s.startswith(".") else "." + s for s in video_suffixes
            ]
            videos += [
                file for file in video_path.iterdir() if file.suffix in video_suffixes
            ]
        else:
            assert (
                video_path.exists()
            ), f"Could not find the video: {video_path}. Check access rights."
            videos.append(video_path)

    return videos


def ensure_multianimal_df_format(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to 'multianimal' format (with an "individuals" columns index)

    Args:
        df_predictions: the dataframe to convert

    Returns:
        the dataframe in MA format
    """
    df_predictions_ma = df_predictions.copy()
    try:
        df_predictions_ma.columns.get_level_values("individuals").unique().tolist()
    except KeyError:
        new_cols = pd.MultiIndex.from_tuples(
            [(col[0], "animal", col[1], col[2]) for col in df_predictions_ma.columns],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df_predictions_ma.columns = new_cols
    return df_predictions_ma


def build_predictions_dataframe(
    scorer: str,
    predictions: dict[str, dict[str, np.ndarray]],
    parameters: PoseDatasetParameters,
    image_name_to_index: Callable[[str], tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """

    Args:
        scorer:
        predictions
        parameters:
        image_name_to_index:

    Returns:

    """
    prediction_data = []
    index_data = []
    for image, image_predictions in predictions.items():
        image_data = image_predictions["bodyparts"][..., :3].reshape(-1)
        if "unique_bodyparts" in image_predictions:
            image_data = np.concatenate(
                [image_data, image_predictions["unique_bodyparts"][..., :3].reshape(-1)]
            )

        prediction_data.append(image_data)
        if image_name_to_index is not None:
            index_data.append(image_name_to_index(image))

    if len(index_data) > 0:
        index = pd.MultiIndex.from_tuples(index_data)
    else:
        index = list(predictions.keys())

    return pd.DataFrame(
        prediction_data,
        index=index,
        columns=build_dlc_dataframe_columns(
            scorer=scorer,
            parameters=parameters,
            with_likelihood=True,
        ),
    )


def get_inference_runners(
    model_config: dict,
    snapshot_path: str | Path,
    max_individuals: int,
    num_bodyparts: int,
    num_unique_bodyparts: int,
    device: str | None = None,
    with_identity: bool = False,
    transform: A.BaseCompose | None = None,
    detector_path: str | Path | None = None,
    detector_transform: A.BaseCompose | None = None,
) -> tuple[InferenceRunner, InferenceRunner | None]:
    """Builds the runners for pose estimation

    Args:
        model_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image
        num_bodyparts: the number of bodyparts predicted by the model
        num_unique_bodyparts: the number of unique_bodyparts predicted by the model
        with_identity: whether the pose model has an identity head
        device: if defined, overwrites the device selection from the model config
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.
        detector_path: the path to the detector snapshot from which to load weights,
            for top-down models (if a detector runner is needed)
        detector_transform: the transform for object detection. if None, uses the
            transform defined in the config.

    Returns:
        a runner for pose estimation
        a runner for detection, if detector_path is not None
    """
    pose_task = Task(model_config["method"])
    if device is None:
        device = resolve_device(model_config)

    if transform is None:
        transform = build_transforms(model_config["data"]["inference"])

    detector_runner = None
    if pose_task == Task.BOTTOM_UP:
        pose_preprocessor = build_bottom_up_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
        )
        pose_postprocessor = build_bottom_up_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
            with_identity=with_identity,
        )
    else:
        # FIXME: Cannot run detectors on MPS
        detector_device = device
        if device == "mps":
            detector_device = "cpu"

        pose_preprocessor = build_top_down_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
            cropped_image_size=(256, 256),
        )
        pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
        )

        if detector_path is not None:
            if detector_transform is None:
                detector_transform = build_transforms(
                    model_config["detector"]["data"]["inference"]
                )

            detector_config = model_config["detector"]["model"]
            if "pretrained" in detector_config:
                detector_config["pretrained"] = False

            detector_runner = build_inference_runner(
                task=Task.DETECT,
                model=DETECTORS.build(detector_config),
                device=detector_device,
                snapshot_path=detector_path,
                preprocessor=build_bottom_up_preprocessor(
                    color_mode=model_config["detector"]["data"]["colormode"],
                    transform=detector_transform,
                ),
                postprocessor=build_detector_postprocessor(
                    max_individuals=max_individuals,
                ),
            )

    pose_runner = build_inference_runner(
        task=pose_task,
        model=PoseModel.build(model_config["model"]),
        device=device,
        snapshot_path=snapshot_path,
        preprocessor=pose_preprocessor,
        postprocessor=pose_postprocessor,
    )
    return pose_runner, detector_runner
