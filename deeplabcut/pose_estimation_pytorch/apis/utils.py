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
import random
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import pandas as pd

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.data.ctd import CondFromModel
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
    build_conditional_top_down_preprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.runners import (
    build_inference_runner,
    CTDTrackingConfig,
    DetectorInferenceRunner,
    DynamicCropper,
    InferenceRunner,
    PoseInferenceRunner,
    TopDownDynamicCropper,
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
    snapshot_filter: list[str] | None = None,
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
        snapshot_filter: List of snapshot names to return (e.g. ["snapshot-50",
            "snapshot-75"]). If defined, `index` will be ignored.

    Returns:
        If index=="all", returns all snapshots. Otherwise, returns a list containing a
        single snapshot, with the desired index.

    Raises:
        ValueError: If the index given is not valid
        ValueError: If index=="best" but there is no saved best model
    """
    snapshot_manager = TorchSnapshotManager(
        model_folder=model_folder, snapshot_prefix=task.snapshot_prefix
    )
    if snapshot_filter is not None:
        all_snapshots = snapshot_manager.snapshots()
        snapshots = [s for s in all_snapshots if s.path.stem in snapshot_filter]
        if len(snapshots) != len(snapshot_filter):
            print(f"Warning: could not find all `snapshots_to_evaluate`.")
            print(f"  Requested snapshots: {snapshot_filter}")
            print(f"  Found snapshots: {[s.path.stem for s in all_snapshots]}")
            print(f"  Snapshots returned: {[s.path.stem for s in snapshots]}")
        return snapshots

    if isinstance(index, str) and index.lower() == "best":
        best_snapshot = snapshot_manager.best()
        if best_snapshot is None:
            raise ValueError(f"No best snapshot found in {model_folder}")
        snapshots = [best_snapshot]
    elif isinstance(index, str) and index.lower() == "all":
        snapshots = snapshot_manager.snapshots()
    elif isinstance(index, int):
        all_snapshots = snapshot_manager.snapshots()
        if (
            len(all_snapshots) == 0
            or len(all_snapshots) <= index
            or (index < 0 and len(all_snapshots) < -index)
        ):
            names = [s.path.name for s in all_snapshots]
            raise ValueError(
                f"Found {len(all_snapshots)} snapshots in {model_folder} (with names "
                f"{names}) with prefix {snapshot_manager.snapshot_prefix}. Could "
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
            try:
                detector_snapshot = get_model_snapshots(
                    detector_index, train_dir, Task.DETECT
                )[0]
            except ValueError:
                detector_snapshot = None

        snapshot_uid = get_scorer_uid(snapshot, detector_snapshot)

    task, date = cfg["Task"], cfg["date"]
    name = "".join([p.capitalize() for p in net_type.split("_")])
    return f"DLC_{name}_{task}{date}shuffle{shuffle}_{snapshot_uid}"


def list_videos_in_folder(
    data_path: str | list[str],
    video_type: str | None,
    shuffle: bool = False,
) -> list[Path]:
    """
    Args:
        data_path: Path or list of paths to folders containing videos
        video_type: The type of video to filter for
        shuffle: If the paths point to directories, whether to shuffle the order of
            videos in the directory.

    Returns:
        The paths of videos to analyze.
    """
    if not isinstance(data_path, list):
        data_path = [data_path]
    video_paths = [Path(p) for p in data_path]

    videos = []
    for path in video_paths:
        if path.is_dir():
            if not video_type:
                video_suffixes = ["." + ext for ext in auxfun_videos.SUPPORTED_VIDEOS]
            else:
                video_suffixes = [video_type]

            suffixes = [s if s.startswith(".") else "." + s for s in video_suffixes]
            videos_in_dir = [file for file in path.iterdir() if file.suffix in suffixes]
            if shuffle:
                random.shuffle(videos_in_dir)
            videos += videos_in_dir
        else:
            assert (
                path.exists()
            ), f"Could not find the video: {path}. Check access rights."
            videos.append(path)

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


def _image_names_to_df_index(
    image_names: list[str],
    image_name_to_index: Callable[[str], tuple[str, ...]] | None = None,
) -> pd.MultiIndex | list[str]:
    """
    Creates index for predictions dataframe.
    This method is used in build_predictions_dataframe, but also in build_bboxes_dict_for_dataframe.
    It is important that these two methods return objects with the same index / keys.

    Args:
        image_names: list of image names
        image_name_to_index, optional: a transform to apply on each image_name
    """

    if image_name_to_index is not None:
        return pd.MultiIndex.from_tuples(
            [image_name_to_index(image_name) for image_name in image_names]
        )
    else:
        return image_names


def build_predictions_dataframe(
    scorer: str,
    predictions: dict[str, dict[str, np.ndarray]],
    parameters: PoseDatasetParameters,
    image_name_to_index: Callable[[str], tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Builds a pandas DataFrame from pose prediction data. The resulting DataFrame
    includes properly formatted indices and column names for compatibility with
    DeepLabCut workflows.

    Args:
        scorer: The name of the scorer used to generate the predictions.
        predictions: A dictionary where each key is an image name and its value is
            another dictionary. The inner dictionary contains prediction data for
            "bodyparts" and optionally "unique_bodyparts". The "bodyparts" and
            "unique_bodyparts" data arrays are expected to be 3-dimensional, containing
            pose predictions in format (num_predicted_individuals, num_bodyparts, 3).
        parameters: Dataset-specific parameters required for constructing DataFrame
            columns.
        image_name_to_index: A callable function that takes an image name and returns
            a tuple representing the DataFrame index. If None, indices will be
            generated without transformation.

    Returns:
        A pandas DataFrame containing the processed prediction data for all provided
        images. The DataFrame index corresponds to the image names or their
        transformed values (if `image_name_to_index` is provided). The DataFrame
        columns are constructed using the provided scorer and parameters.
    """
    image_names = []
    prediction_data = []
    
    # Check if this is a humanbody model by looking at the first prediction
    if predictions:
        first_pred = next(iter(predictions.values()))
        if "bodyparts" in first_pred:
            actual_num_individuals = first_pred["bodyparts"].shape[0]
            expected_num_individuals = len(parameters.individuals)
            
            # For humanbody models, if the actual number of individuals differs from expected,
            # we need to adjust the parameters to match the actual predictions
            if actual_num_individuals != expected_num_individuals:
                # Create adjusted parameters with the actual number of individuals
                adjusted_individuals = [f"individual_{i}" for i in range(actual_num_individuals)]
                parameters = PoseDatasetParameters(
                    bodyparts=parameters.bodyparts,
                    unique_bpts=parameters.unique_bpts,
                    individuals=adjusted_individuals,
                )
    
    for image_name, image_predictions in predictions.items():
        image_data = image_predictions["bodyparts"][..., :3].reshape(-1)
        if "unique_bodyparts" in image_predictions:
            image_data = np.concatenate(
                [image_data, image_predictions["unique_bodyparts"][..., :3].reshape(-1)]
            )
        image_names.append(image_name)
        prediction_data.append(image_data)

    index = _image_names_to_df_index(image_names, image_name_to_index)

    return pd.DataFrame(
        prediction_data,
        index=index,
        columns=build_dlc_dataframe_columns(
            scorer=scorer,
            parameters=parameters,
            with_likelihood=True,
        ),
    )


def build_bboxes_dict_for_dataframe(
    predictions: dict[str, dict[str, np.ndarray]],
    image_name_to_index: Callable[[str], tuple[str, ...]] | None = None,
) -> dict:
    """
    Creates a dictionary with bounding boxes from predictions.

    The keys of the dictionary are the same as the index of the dataframe created by
    build_predictions_dataframe. Therefore, the structures returned by
    build_predictions_dataframe and by build_bboxes_dict_for_dataframe can be accessed
    with the same keys.

    Args:
        predictions: Dictionary containing the evaluation results
        image_name_to_index: a transform to apply on each image_name

    Returns:
        Dictionary with sames keys as in the dataframe returned by
        build_predictions_dataframe, and respective bounding boxes and scores, if any.
    """

    image_names = []
    bboxes_data = []
    for image_name, image_predictions in predictions.items():
        image_names.append(image_name)
        if "bboxes" in image_predictions and "bbox_scores" in image_predictions:
            bboxes_data.append(
                (image_predictions["bboxes"], image_predictions["bbox_scores"])
            )

    index = _image_names_to_df_index(image_names, image_name_to_index)

    return dict(zip(index, bboxes_data))


def get_inference_runners(
    model_config: dict,
    snapshot_path: str | Path,
    max_individuals: int | None = None,
    num_bodyparts: int | None = None,
    num_unique_bodyparts: int | None = None,
    batch_size: int = 1,
    device: str | None = None,
    with_identity: bool = False,
    transform: A.BaseCompose | None = None,
    detector_batch_size: int = 1,
    detector_path: str | Path | None = None,
    detector_transform: A.BaseCompose | None = None,
    dynamic: DynamicCropper | None = None,
) -> tuple[InferenceRunner, InferenceRunner | None]:
    """Builds the runners for pose estimation

    Args:
        model_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image (if None, uses the
            individuals defined in the model_config metadata)
        num_bodyparts: the number of bodyparts predicted by the model (if None, uses the
            bodyparts defined in the model_config metadata)
        num_unique_bodyparts: the number of unique_bodyparts predicted by the model (if
            None, uses the unique bodyparts defined in the model_config metadata)
        batch_size: the batch size to use for the pose model.
        with_identity: whether the pose model has an identity head
        device: if defined, overwrites the device selection from the model config
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.
        detector_batch_size: the batch size to use for the detector
        detector_path: the path to the detector snapshot from which to load weights,
            for top-down models (if a detector runner is needed)
        detector_transform: the transform for object detection. if None, uses the
            transform defined in the config.
        dynamic: The DynamicCropper used for video inference, or None if dynamic
            cropping should not be used. Only for bottom-up pose estimation models.
            Should only be used when creating inference runners for video pose
            estimation with batch size 1.

    Returns:
        a runner for pose estimation
        a runner for detection, if detector_path is not None
    """
    if max_individuals is None:
        max_individuals = len(model_config["metadata"]["individuals"])
    if num_bodyparts is None:
        num_bodyparts = len(model_config["metadata"]["bodyparts"])
    if num_unique_bodyparts is None:
        num_unique_bodyparts = len(model_config["metadata"]["unique_bodyparts"])

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
        crop_cfg = model_config["data"]["inference"].get("top_down_crop", {})
        width, height = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
        margin = crop_cfg.get("margin", 0)
        if pose_task == Task.COND_TOP_DOWN:
            pose_preprocessor = build_conditional_top_down_preprocessor(
                color_mode=model_config["data"]["colormode"],
                transform=transform,
                bbox_margin=model_config["data"].get("bbox_margin", 20),
                top_down_crop_size=(width, height),
                top_down_crop_margin=margin,
                top_down_crop_with_context=crop_cfg.get("crop_with_context", False),
            )
        else:  # Top-Down
            pose_preprocessor = build_top_down_preprocessor(
                color_mode=model_config["data"]["colormode"],
                transform=transform,
                top_down_crop_size=(width, height),
                top_down_crop_margin=margin,
                top_down_crop_with_context=crop_cfg.get("crop_with_context", True),
            )

        pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique_bodyparts,
        )

        # FIXME: Cannot run detectors on MPS
        detector_device = device
        if device == "mps":
            detector_device = "cpu"

        # Get superanimal name for filtering logic
        superanimal_name = model_config.get("metadata", {}).get("superanimal_name", "")

        if detector_path is not None or "detector" in model_config:
            if detector_path is not None:
                detector_path = str(detector_path)
            if detector_transform is None:
                detector_transform = build_transforms(
                    model_config["detector"]["data"]["inference"]
                )

            print(f"DEBUG: Creating detector for superanimal_name: '{superanimal_name}'")
            if superanimal_name == "superanimal_humanbody":
                # Only for superanimal_humanbody, use torchvision detector
                from deeplabcut.pose_estimation_pytorch.models.detectors.torchvision import TorchvisionDetectorAdaptor
                detector_config = model_config["detector"]["model"].copy()
                expected_fields = {
                    "model", "weights", "num_classes", "freeze_bn_stats", "freeze_bn_weights", 
                    "box_score_thresh", "model_kwargs", "model_name", "superanimal_name"
                }
                unexpected_fields = [k for k in detector_config.keys() if k not in expected_fields]
                for field in unexpected_fields:
                    detector_config.pop(field, None)
                if detector_path is not None:
                    detector_config["weights"] = None
                detector_model = TorchvisionDetectorAdaptor(**detector_config)
                detector_model.superanimal_name = superanimal_name
                print(f"DEBUG: Created TorchvisionDetectorAdaptor for {superanimal_name}")
            else:
                # For all other superanimal models, use the original logic (pre-humanbody integration)
                detector_config = model_config["detector"]["model"].copy()
                pretrained = False if detector_path is not None else True
                detector_model = DETECTORS.build(detector_config, pretrained=pretrained)
                detector_model.superanimal_name = superanimal_name
                print(f"DEBUG: Created custom detector from DETECTORS registry for {superanimal_name}")
                print(f"DEBUG: Custom detector type: {type(detector_model)}")
            detector_runner = build_inference_runner(
                task=Task.DETECT,
                model=detector_model,
                device=detector_device,
                snapshot_path=detector_path,
                batch_size=detector_batch_size,
                preprocessor=build_bottom_up_preprocessor(
                    color_mode=model_config["detector"]["data"]["colormode"],
                    transform=detector_transform,
                ),
                postprocessor=build_detector_postprocessor(
                    max_individuals=max_individuals,
                ),
                load_weights_only=model_config["detector"]["runner"].get(
                    "load_weights_only",
                    None,
                ),
            )

    pose_runner = build_inference_runner(
        task=pose_task,
        model=PoseModel.build(model_config["model"]),
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=pose_preprocessor,
        postprocessor=pose_postprocessor,
        dynamic=dynamic,
        load_weights_only=model_config["runner"].get("load_weights_only", None),
    )
    return pose_runner, detector_runner


def get_detector_inference_runner(
    model_config: dict,
    snapshot_path: str | Path,
    batch_size: int = 1,
    device: str | None = None,
    max_individuals: int | None = None,
    transform: A.BaseCompose | None = None,
) -> DetectorInferenceRunner:
    """Builds an inference runner for object detection.

    Args:
        model_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image
        batch_size: the batch size to use for the pose model.
        device: if defined, overwrites the device selection from the model config
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.

    Returns:
        an inference runner for object detection
    """
    if device is None:
        device = resolve_device(model_config)
    elif device == "mps":  # FIXME(niels): Cannot run detectors on MPS
        device = "cpu"

    if max_individuals is None:
        max_individuals = len(model_config["metadata"]["individuals"])

    det_cfg = model_config["detector"]
    if transform is None:
        transform = build_transforms(det_cfg["data"]["inference"])

    if "pretrained" in det_cfg["model"]:
        det_cfg["model"]["pretrained"] = False

    preprocessor = build_bottom_up_preprocessor(det_cfg["data"]["colormode"], transform)
    postprocessor = build_detector_postprocessor(max_individuals=max_individuals)
    runner = build_inference_runner(
        task=Task.DETECT,
        model=DETECTORS.build(det_cfg["model"]),
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        load_weights_only=det_cfg["runner"].get("load_weights_only", None),
    )

    if not isinstance(runner, DetectorInferenceRunner):
        raise RuntimeError(f"Failed to build DetectorInferenceRunner: {model_config}")

    return runner


def get_pose_inference_runner(
    model_config: dict,
    snapshot_path: str | Path,
    batch_size: int = 1,
    device: str | None = None,
    max_individuals: int | None = None,
    transform: A.BaseCompose | None = None,
    dynamic: DynamicCropper | None = None,
    cond_provider: CondFromModel | None = None,
    ctd_tracking: bool | CTDTrackingConfig = False,
) -> PoseInferenceRunner:
    """Builds an inference runner for pose estimation.

    Args:
        model_config: the pytorch configuration file
        snapshot_path: the path of the snapshot from which to load the weights
        max_individuals: the maximum number of individuals per image
        batch_size: the batch size to use for the pose model.
        device: if defined, overwrites the device selection from the model config
        transform: the transform for pose estimation. if None, uses the transform
            defined in the config.
        dynamic: The DynamicCropper used for video inference, or None if dynamic
            cropping should not be used. Should only be used when creating inference
            runners for video pose estimation with batch size 1. For top-down pose
            estimation models, a `TopDownDynamicCropper` must be used.
        cond_provider: Only for CTD models. If None, the CondProvider is created from
            the pytorch_cfg.
        ctd_tracking: Only for CTD models. Conditional top-down models can be used
            to directly track individuals. Poses from frame T are given as conditions
            for frame T+1. This also means a BU model is only needed to "initialize" the
            pose in the first frame, and for the remaining frames only the CTD model is
            needed. To configure conditional pose tracking differently, you can pass a
            CTDTrackingConfig instance.

    Returns:
        an inference runner for pose estimation
    """
    pose_task = Task(model_config["method"])
    metadata = model_config["metadata"]
    num_bodyparts = len(metadata["bodyparts"])
    num_unique = len(metadata["unique_bodyparts"])
    with_identity = bool(metadata["with_identity"])
    if max_individuals is None:
        max_individuals = len(metadata["individuals"])

    if device is None:
        device = resolve_device(model_config)

    if transform is None:
        transform = build_transforms(model_config["data"]["inference"])

    kwargs = {}
    if pose_task == Task.BOTTOM_UP or isinstance(dynamic, TopDownDynamicCropper):
        pose_preprocessor = build_bottom_up_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
        )
        pose_postprocessor = build_bottom_up_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique,
            with_identity=with_identity,
        )
    else:
        crop_cfg = model_config["data"]["inference"].get("top_down_crop", {})
        width, height = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
        margin = crop_cfg.get("margin", 0)

        if pose_task == Task.COND_TOP_DOWN:
            if cond_provider is not None:
                kwargs["bu_runner"] = get_pose_inference_runner(
                    model_config=read_config_as_dict(cond_provider.config_path),
                    snapshot_path=cond_provider.snapshot_path,
                    batch_size=1,
                    device=device,
                    max_individuals=max_individuals,
                )

            kwargs["ctd_tracking"] = ctd_tracking

            pose_preprocessor = build_conditional_top_down_preprocessor(
                color_mode=model_config["data"]["colormode"],
                transform=transform,
                bbox_margin=model_config["data"].get("bbox_margin", 20),
                top_down_crop_size=(width, height),
                top_down_crop_margin=margin,
                top_down_crop_with_context=crop_cfg.get("crop_with_context", False),
            )
        else:  # Top-Down
            pose_preprocessor = build_top_down_preprocessor(
                color_mode=model_config["data"]["colormode"],
                transform=transform,
                top_down_crop_size=(width, height),
                top_down_crop_margin=margin,
                top_down_crop_with_context=crop_cfg.get("crop_with_context", True),
            )

        pose_postprocessor = build_top_down_postprocessor(
            max_individuals=max_individuals,
            num_bodyparts=num_bodyparts,
            num_unique_bodyparts=num_unique,
        )

    runner = build_inference_runner(
        task=pose_task,
        model=PoseModel.build(model_config["model"]),
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=pose_preprocessor,
        postprocessor=pose_postprocessor,
        dynamic=dynamic,
        load_weights_only=model_config["runner"].get("load_weights_only", None),
        **kwargs,
    )
    if not isinstance(runner, PoseInferenceRunner):
        raise RuntimeError(f"Failed to build PoseInferenceRunner for {model_config}")

    return runner
