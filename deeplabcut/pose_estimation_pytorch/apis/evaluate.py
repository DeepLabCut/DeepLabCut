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

import argparse
from pathlib import Path
from typing import Iterable

import albumentations as A
import numpy as np
import pandas as pd

import deeplabcut.pose_estimation_pytorch as dlc
import deeplabcut.pose_estimation_pytorch.runners.utils as runner_utils
from deeplabcut.pose_estimation_pytorch import Loader, DLCLoader
from deeplabcut.pose_estimation_pytorch.apis.scoring import (
    align_predicted_individuals_to_gt,
    get_scores,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_predictions_dataframe,
    ensure_multianimal_df_format,
    get_runners,
)
from deeplabcut.pose_estimation_pytorch.runners import Runner
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.visualization import plot_evaluation_results


def predict(
    task: str,
    pose_runner: Runner,
    loader: Loader,
    mode: str,
    detector_runner: Runner | None = None,
) -> tuple[list[str], list[dict[str, dict[str, np.ndarray]]]]:
    """Predicts poses on data contained in a loader
    Args:
        task: {'TD', 'BU'} Whether the model is a top-down or bottom-up model
        pose_runner: The runner to use for pose estimation
        loader: The loader containing the data to predict poses on
        mode: {"train", "test"} The mode to predict on
        detector_runner: If the task is "TD", a detector runner can be given to detect
            individuals in the images. If no detector is given, ground truth bounding
            boxes will be used to crop individuals before pose estimation

    Returns:
        The paths of images for which predictions were computed
        For each image, the predictions made by each model head
    """
    if task not in ["BU", "TD"]:
        raise ValueError(
            f"Task should be set to either 'BU' (Bottom Up) or 'TD' (Top Down), "
            f"currently it is {task}"
        )

    image_paths = loader.image_filenames(mode)
    context = None

    if task == "TD":
        # Get bounding boxes for context
        if detector_runner is not None:
            bbox_predictions = detector_runner.inference(images=image_paths)
            context = bbox_predictions
        else:
            ground_truth_bboxes = loader.ground_truth_bboxes(mode=mode)
            context = [{"bboxes": ground_truth_bboxes[image]} for image in image_paths]

    images = image_paths
    if context is not None:
        if len(context) != len(image_paths):
            raise ValueError(
                f"Missing context for some images: {len(context)} != {len(image_paths)}"
            )
        images = list(zip(image_paths, context))

    predictions = pose_runner.inference(images=images)
    return image_paths, predictions  # TODO: include bounding boxes if there are any


def evaluate(
    scorer: str,
    task: str,
    pose_runner: Runner,
    loader: Loader,
    mode: str,
    detector_runner: Runner | None = None,
    pcutoff: float = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Args:
        scorer: The name of the model making the predictions
        task: {'BU' or 'TD'} Whether to run top-down or bottom-up
        pose_runner: The runner for pose estimation
        loader: The loader containing the data to evaluate
        mode: Either 'train' or 'test'
        detector_runner: If task == 'TD', a detector can be given to compute bounding
            boxes for pose estimation. If no detector is given, ground truth bounding
            boxes are used
        pcutoff: The p-cutoff to use for evaluation

    Returns:
        A dict containing the evaluation results
        A dataframe in DLC-format containing the predictions
    """
    parameters = loader._get_dataset_parameters()
    image_paths, predictions = predict(
        task=task,
        pose_runner=pose_runner,
        loader=loader,
        mode=mode,
        detector_runner=detector_runner,
    )
    # TODO: move this to postprocessing step
    poses = {}
    for filename, pred in zip(image_paths, predictions):
        keypoints = pred["bodyparts"]
        if len(keypoints) < parameters.max_num_animals:
            padded_keypoints = np.empty(
                (parameters.max_num_animals, *keypoints.shape[1:])
            )
            padded_keypoints.fill(-1)
            padded_keypoints[: len(keypoints), ...] = keypoints
            keypoints = padded_keypoints
        poses[filename] = keypoints

    unique_poses = None
    gt_unique_keypoints = None
    if parameters.num_unique_bpts > 1:
        unique_poses = {
            filename: pred["unique_bodyparts"]
            for filename, pred in zip(image_paths, predictions)
        }
        gt_unique_keypoints = loader.ground_truth_keypoints(mode, unique_bodypart=True)

    gt_keypoints = loader.ground_truth_keypoints(mode)
    if parameters.max_num_animals > 1:
        poses = align_predicted_individuals_to_gt(poses, gt_keypoints)

    # TODO: Check single animal mAP computation
    results = get_scores(
        poses,
        gt_keypoints,
        pcutoff=pcutoff,
        unique_bodypart_poses=unique_poses,
        unique_bodypart_gt=gt_unique_keypoints,
    )

    image_name_to_index = None
    if isinstance(loader, DLCLoader):
        image_name_to_index = image_to_dlc_df_index

    df_predictions = build_predictions_dataframe(
        scorer=scorer,
        images=image_paths,
        bodypart_predictions=poses,
        unique_bodypart_predictions=unique_poses,
        parameters=parameters,
        image_name_to_index=image_name_to_index,
    )
    return results, df_predictions


def evaluate_snapshot(
    cfg: dict,
    shuffle: int = 0,
    trainingsetindex: int = -1,
    snapshotindex: int = -1,
    device: str | None = None,
    transform: A.Compose | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    modelprefix: str = "",
    detector_path: str | None = None,
) -> pd.DataFrame:
    """Evaluates a snapshot.
    The evaluation results are stored in the .h5 and TODO .csv file under the subdirectory
    'evaluation_results'.

    Args:
        cfg: the content of the project's config file
        shuffle: shuffle index
        trainingsetindex: the training set fraction to use
        modelprefix: model prefix
        snapshotindex: index (starting at 0) of the snapshot we want to load. To
            evaluate the last one, use -1. To evaluate all snapshots, use "all". For
            example if we have 3 models saved
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshotindex is loaded from the project configuration.
        device: the device to run evaluation on
        transform: transformation pipeline for evaluation
            ** Should normalise the data the same way it was normalised during training **
        plotting: Plots the predictions on the train and test images. If provided it must
            be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
            to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: whether to compare predictions and ground truth
        detector_path: Only for TD models. If defined, evaluation metrics are computed
            using the detections made by this detector
    """
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    model_folder = runner_utils.get_model_folder(
        cfg["project_path"],
        cfg,
        train_fraction,
        shuffle,
        modelprefix,
    )
    model_config_path = str(Path(model_folder) / "train" / "pytorch_config.yaml")
    pytorch_config = auxiliaryfunctions.read_plainconfig(model_config_path)
    if device is not None:
        pytorch_config["device"] = device

    method = pytorch_config.get("method", "bu").lower()
    if method not in ["bu", "td"]:
        raise ValueError(
            f"Method should be set to either 'bu' (Bottom Up) or 'td' (Top Down), "
            f"currently it is {method}"
        )

    loader = dlc.DLCLoader(
        project_root=pytorch_config["project_path"],
        model_config_path=model_config_path,
        shuffle=shuffle,
    )
    parameters = loader._get_dataset_parameters()
    names = runner_utils.get_paths(
        project_path=cfg["project_path"],
        train_fraction=train_fraction,
        model_prefix=modelprefix,
        shuffle=shuffle,
        cfg=cfg,
        train_iterations=snapshotindex,
        method=method,
    )
    pcutoff = cfg.get("pcutoff")

    pose_runner, detector_runner = get_runners(
        pytorch_config=pytorch_config,
        snapshot_path=names["model_path"],
        with_unique_bodyparts=(parameters.num_unique_bpts > 0),
        transform=transform,
        detector_path=detector_path,
        detector_transform=None,
    )

    predictions = {}
    scores = {
        "Training epochs": int(names["dlc_scorer"].split("_")[-1]),
        "%Training dataset": train_fraction,
        "Shuffle number": shuffle,
        "pcutoff": pcutoff,
    }
    for split in ["train", "test"]:
        results, df_split_predictions = evaluate(
            scorer=names["dlc_scorer"],
            task=pytorch_config.get("method", "BU").upper(),
            pose_runner=pose_runner,
            loader=loader,
            mode=split,
            pcutoff=pcutoff,
            detector_runner=detector_runner,
        )
        predictions[split] = df_split_predictions
        for k, v in results.items():
            scores[f"{split} {k}"] = round(v, 2)

    results_filename = runner_utils.get_results_filename(
        names["evaluation_folder"],
        names["dlc_scorer"],
        names["dlc_scorer_legacy"],
        names["model_path"][:-3],
    )
    df_predictions = pd.concat(predictions.values(), axis=0)
    df_predictions = df_predictions.reindex(loader.df_dlc.index)
    output_filename = Path(results_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_hdf(str(output_filename), "df_with_missing")

    df_scores = pd.DataFrame([scores]).set_index(
        ["Training epochs", "%Training dataset", "Shuffle number", "pcutoff"]
    )
    scores_filepath = Path(results_filename).with_suffix(".csv")
    scores_filepath = scores_filepath.with_stem(scores_filepath.stem + "-results")
    save_evaluation_results(df_scores, scores_filepath, show_errors, pcutoff)

    if plotting:
        snapshot_name = Path(names["model_path"]).stem
        folder_name = (
            f"{names['evaluation_folder']}/"
            f"LabeledImages_{names['dlc_scorer']}_{snapshot_name}"
        )
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        if isinstance(plotting, str):
            plot_mode = plotting
        else:
            plot_mode = "bodypart"

        df_ground_truth = ensure_multianimal_df_format(loader.df_dlc)
        for mode in ["train", "test"]:
            df_combined = predictions[mode].merge(
                df_ground_truth, left_index=True, right_index=True
            )
            plot_unique_bodyparts = False  # TODO: get from parameters
            plot_evaluation_results(
                df_combined=df_combined,
                project_root=cfg["project_path"],
                scorer=cfg["scorer"],
                model_name=names["dlc_scorer"],
                output_folder=folder_name,
                in_train_set=mode == "train",
                plot_unique_bodyparts=plot_unique_bodyparts,
                mode=plot_mode,
                colormap=cfg["colormap"],
                dot_size=cfg["dotsize"],
                alpha_value=cfg["alphavalue"],
                p_cutoff=cfg["pcutoff"],
            )

    return df_predictions


def evaluate_network(
    config: str,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    transform: A.Compose = None,
    modelprefix: str = "",
    detector_path: str | None = None,
) -> None:
    """Evaluates a snapshot.

    The evaluation results are stored in the .h5 and .csv file under the subdirectory
    'evaluation_results'.

    Args:
        config: path to the project's config file
        shuffles: Iterable of integers specifying the shuffle indices to evaluate.
        trainingsetindex: Integer specifying which training set fraction to use.
            Evaluates all fractions if set to "all"
        snapshotindex: index (starting at 0) of the snapshot we want to load. To
            evaluate the last one, use -1. To evaluate all snapshots, use "all". For
            example if we have 3 models saved
                - snapshot-0.pt
                - snapshot-50.pt
                - snapshot-100.pt
            and we want to evaluate snapshot-50.pt, snapshotindex should be 1. If None,
            the snapshotindex is loaded from the project configuration.
        device: the device to run evaluation on
        plotting: Plots the predictions on the train and test images. If provided it must
            be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
            to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: display train and test errors.
        transform: transformation pipeline for evaluation
            ** Should normalise the data the same way it was normalised during training **
        modelprefix: directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        detector_path: Only for TD models. If defined, evaluation metrics are computed
            using the detections made by this detector

    Examples:
        If you want to evaluate on shuffle 1 without plotting predictions.

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml', shuffles=[1],
            )

        If you want to evaluate shuffles 0 and 1 and plot the predictions.

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml',
                shuffles=[0, 1],
                plotting=True,
            )

        If you want to plot assemblies for a maDLC project

        >>> deeplabcut.evaluate_network(
                '/analysis/project/reaching-task/config.yaml',
                shuffles=[1],
                plotting="individual",
            )
    """
    cfg = auxiliaryfunctions.read_config(config)

    if isinstance(trainingsetindex, int):
        train_set_indices = [trainingsetindex]
    elif isinstance(trainingsetindex, str) and trainingsetindex.lower() == "all":
        train_set_indices = list(range(len(cfg["TrainingFraction"])))
    else:
        raise ValueError(f"Invalid trainingsetindex: {trainingsetindex}")

    if snapshotindex is None:
        snapshotindex = cfg["snapshotindex"]

    for train_set_index in train_set_indices:
        for shuffle in shuffles:
            if isinstance(snapshotindex, str) and snapshotindex.lower() == "all":
                model_folder = runner_utils.get_model_folder(
                    project_path=str(Path(config).parent),
                    cfg=cfg,
                    train_fraction=cfg["TrainingFraction"][train_set_index],
                    shuffle=shuffle,
                    model_prefix=modelprefix,
                )
                all_snapshots = runner_utils.get_snapshots(Path(model_folder))
                snapshot_indices = list(range(len(all_snapshots)))
            elif isinstance(snapshotindex, int):
                snapshot_indices = [snapshotindex]
            else:
                raise ValueError(f"Invalid snapshotindex: {snapshotindex}")

            for snapshot in snapshot_indices:
                _ = evaluate_snapshot(
                    cfg=cfg,
                    shuffle=shuffle,
                    trainingsetindex=train_set_index,
                    snapshotindex=snapshot,
                    device=device,
                    transform=transform,
                    plotting=plotting,
                    show_errors=show_errors,
                    modelprefix=modelprefix,
                    detector_path=detector_path,
                )


def image_to_dlc_df_index(image: str) -> tuple[str, ...]:
    """
    Args:
        image: the path of the image to map to a DLC index

    Returns:
        the image index to create a multi-animal DLC dataframe:
            ("labeled-data", video_name, image_name)
    """
    image_path = Path(image)
    if len(image_path.parts) >= 3 and image_path.parts[-3] == "labeled-data":
        return Path(image_path).parts[-3:]

    raise ValueError(f"Unexpected image filepath for a DLC project")


def save_evaluation_results(
    df_scores: pd.DataFrame,
    scores_path: Path,
    print_results: bool,
    pcutoff: float,
) -> None:
    """
    Saves the evaluation results to a CSV file. Adds the evaluation results for the
    model to the combined results file, or creates it if it does not yet exist.

    Args:
        df_scores: the scores dataframe for a snapshot
        scores_path: the path where the model scores CSV should be saved
        print_results: whether to print evaluation results to the console
        pcutoff: the pcutoff used to get the evaluation results
    """
    if print_results:
        print(f"Evaluation results for {scores_path.name} (pcutoff: {pcutoff}):")
        print(df_scores.iloc[0])

    # Save scores file
    df_scores.to_csv(scores_path)

    # Update combined results
    combined_scores_path = scores_path.parent.parent / "CombinedEvaluation-results.csv"
    if combined_scores_path.exists():
        df_existing_results = pd.read_csv(combined_scores_path, index_col=[0, 1, 2, 3])
        df_scores = df_scores.combine_first(df_existing_results)

    df_scores = df_scores.sort_index()
    df_scores.to_csv(combined_scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--modelprefix", type=str, default="")
    parser.add_argument("--snapshotindex", type=int, default=49)
    parser.add_argument("--plotting", type=bool, default=False)
    parser.add_argument("--show_errors", type=bool, default=True)
    args = parser.parse_args()
    evaluate_network(
        config=args.config,
        modelprefix=args.modelprefix,
        snapshotindex=args.snapshotindex,
        plotting=args.plotting,
        show_errors=args.show_errors,
    )
