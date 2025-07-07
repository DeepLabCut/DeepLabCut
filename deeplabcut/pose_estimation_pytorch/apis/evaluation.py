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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import deeplabcut.core.metrics as metrics
import deeplabcut.pose_estimation_pytorch.apis.ctd as ctd
import deeplabcut.pose_estimation_pytorch.apis.prune_paf_graph as prune_paf_graph
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch import utils
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_predictions_dataframe,
    ensure_multianimal_df_format,
    get_inference_runners,
    get_model_snapshots,
    get_scorer_name,
    get_scorer_uid,
    build_bboxes_dict_for_dataframe,
)
from deeplabcut.pose_estimation_pytorch.data import DLCLoader, Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
from deeplabcut.pose_estimation_pytorch.runners.snapshots import Snapshot
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxfun_videos, auxiliaryfunctions
from deeplabcut.utils.visualization import (
    create_minimal_figure,
    erase_artists,
    get_cmap,
    make_multianimal_labeled_image,
    plot_evaluation_results,
    save_labeled_frame,
)


def predict(
    pose_runner: InferenceRunner,
    loader: Loader,
    mode: str,
    detector_runner: InferenceRunner | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Predicts poses on data contained in a loader

    Args:
        pose_runner: The runner to use for pose estimation
        loader: The loader containing the data to predict poses on
        mode: {"train", "test"} The mode to predict on
        detector_runner: If the loader's `pose_task` is "TD", a detector runner can be
            given to detect individuals in the images. If no detector is given, ground
            truth bounding boxes will be used to crop individuals before pose estimation

    Returns:
        The paths of images for which predictions were computed mapping to the
        different predictions made by each model head
    """
    image_paths = loader.image_filenames(mode)
    context = None

    if loader.pose_task == Task.TOP_DOWN:
        # Get bounding boxes for context
        if detector_runner is not None:
            bbox_predictions = detector_runner.inference(images=tqdm(image_paths))
            context = bbox_predictions
        else:
            ground_truth_bboxes = loader.ground_truth_bboxes(mode=mode)
            context = [
                {"bboxes": ground_truth_bboxes[image]["bboxes"]}
                for image in image_paths
            ]

    elif loader.pose_task == Task.COND_TOP_DOWN:
        # Load conditions for context
        conditions = ctd.load_conditions_for_evaluation(loader, image_paths)
        context = [{"cond_kpts": conditions[image]} for image in image_paths]

    images_with_context = image_paths
    if context is not None:
        if len(context) != len(image_paths):
            raise ValueError(
                f"Missing context for some images: {len(context)} != {len(image_paths)}"
            )
        images_with_context = list(zip(image_paths, context))

    predictions = pose_runner.inference(images=tqdm(images_with_context))
    return {
        image_path: image_predictions
        for image_path, image_predictions in zip(image_paths, predictions)
    }


def evaluate(
    pose_runner: InferenceRunner,
    loader: Loader,
    mode: str,
    detector_runner: InferenceRunner | None = None,
    parameters: PoseDatasetParameters | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    pcutoff: float | list[float] = 0.6,
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]]]:
    """
    Args:
        pose_runner: The runner for pose estimation
        loader: The loader containing the data to evaluate
        mode: Either 'train' or 'test'
        detector_runner: If the loader's `pose_task` is "TD", a detector can be given to
            compute bounding boxes for pose estimation. If no detector is given, ground
            truth bounding boxes are used.
        parameters: PoseDatasetParameters to use. If None, the parameters will be
            obtained from the given Loader. This can be used to change the names of
            bodyparts, e.g. when a model is trained with memory replay.
        comparison_bodyparts: A subset of the bodyparts for which to compute the
            evaluation metrics. Passing "all" or None evaluates on all bodyparts.
        per_keypoint_evaluation: Compute the train and test RMSE for each keypoint, and
            save the results to a {model_name}-keypoint-results.csv in the
            evaluation-results-pytorch folder.
        pcutoff: Confidence threshold for RMSE computation. If a list is provided,
            there should be one value for each bodypart and one value for each unique
            bodypart (if there are any).

    Returns:
        A dict containing the evaluation results
        A dict mapping the paths of images for which predictions were computed to the
            different predictions made by each model head
    """
    predictions = predict(pose_runner, loader, mode, detector_runner=detector_runner)

    # For models trained with memory-replay from SuperAnimal, keep project bodyparts
    if weight_init_cfg := loader.model_cfg["train_settings"].get("weight_init"):
        weight_init = WeightInitialization.from_dict(weight_init_cfg)
        if weight_init.memory_replay:
            for _, pred in predictions.items():
                pred["bodyparts"] = pred["bodyparts"][:, weight_init.conversion_array]

    if parameters is None:
        parameters = loader.get_dataset_parameters()

    gt_pose = loader.ground_truth_keypoints(mode)
    pred_pose = {filename: pred["bodyparts"] for filename, pred in predictions.items()}
    kpt_idx = _get_keypoints_to_use(parameters.bodyparts, comparison_bodyparts)

    gt_unique, pred_unique, unique_idx = None, None, None
    if parameters.num_unique_bpts >= 1:
        gt_unique = loader.ground_truth_keypoints(mode, unique_bodypart=True)
        pred_unique = {
            filename: pred["unique_bodyparts"] for filename, pred in predictions.items()
        }
        unique_idx = _get_keypoints_to_use(parameters.unique_bpts, comparison_bodyparts)

    # When `comparison_bodyparts` is used, check that the bodyparts used for evaluation
    # make sense; If only unique bodyparts are being evaluated, set them as bodyparts
    if kpt_idx is not None and unique_idx is not None:
        if len(kpt_idx) == 0 and len(unique_idx) == 0:
            unique_err = ""
            if len(parameters.unique_bpts) > 0:
                unique_err = f" and the unique_bodyparts are {parameters.unique_bpts}"
            raise ValueError(
                f"No bodyparts left when comparison_bodyparts={comparison_bodyparts}! "
                f"The project bodyparts are {parameters.bodyparts}{unique_err}! Set "
                f"comparison_bodyparts to `None` or `'all'` to evaluate on all of them,"
                f" or select a subset of them to evaluate."
            )
        elif len(kpt_idx) == 0 and len(unique_idx) > 0:
            gt_pose, pred_pose, kpt_idx = gt_unique, pred_unique, unique_idx
            parameters = PoseDatasetParameters(
                bodyparts=parameters.unique_bpts,
                unique_bpts=[],
                individuals=["animal"],
            )
            gt_unique, pred_unique, unique_idx = None, None, None

    if kpt_idx is not None:
        gt_pose = {img: kpts[:, kpt_idx] for img, kpts in gt_pose.items()}
        pred_pose = {img: kpts[:, kpt_idx] for img, kpts in pred_pose.items()}

    if unique_idx is not None:
        gt_unique = {img: kpts[:, unique_idx] for img, kpts in gt_unique.items()}
        pred_unique = {img: kpts[:, unique_idx] for img, kpts in pred_unique.items()}

    bodyparts = _get_subset_bodyparts(parameters.bodyparts, comparison_bodyparts)
    unique_bpts = _get_subset_bodyparts(parameters.unique_bpts, comparison_bodyparts)
    _validate_pcutoff(bodyparts, unique_bpts, pcutoff)

    results = metrics.compute_metrics(
        gt_pose,
        pred_pose,
        single_animal=parameters.max_num_animals == 1,
        pcutoff=pcutoff,
        unique_bodypart_poses=pred_unique,
        unique_bodypart_gt=gt_unique,
        per_keypoint_rmse=per_keypoint_evaluation,
        compute_detection_rmse=False,
    )

    if loader.model_cfg["metadata"]["with_identity"]:
        pred_id_scores = {
            filename: pred["identity_scores"] for filename, pred in predictions.items()
        }
        id_scores = metrics.compute_identity_scores(
            individuals=parameters.individuals,
            bodyparts=parameters.bodyparts,
            predictions=pred_pose,
            identity_scores=pred_id_scores,
            ground_truth=gt_pose,
        )
        for name, score in id_scores.items():
            results[f"id_head_{name}"] = score

    # Updating poses to be aligned and padded
    for image, pose in pred_pose.items():
        predictions[image]["bodyparts"] = pose

    return results, predictions


def visualize_predictions(
    predictions: dict,
    ground_truth: dict,
    output_dir: str | Path | None = None,
    num_samples: int | None = None,
    random_select: bool = False,
    show_ground_truth: bool = True,
    plot_bboxes: bool = True,
) -> None:
    """Visualize model predictions alongside ground truth keypoints.

    This function processes keypoint predictions and ground truth data, applies
    visibility masks, and generates visualization plots. It supports random or
    sequential sampling of images for visualization.

    Args:
        predictions: Dictionary mapping image paths to prediction data.
            Each prediction contains:
            - bodyparts: array of shape [N, num_keypoints, 3] where 3 represents
                (x, y, confidence)
            - bboxes: array of shape [N, 4] for bounding boxes (optional)
            - bbox_scores: array of shape [N,] for bbox confidences (optional)
        ground_truth: Dictionary mapping image paths to ground truth keypoints.
            Each value has shape [N, num_keypoints, 3] where 3 represents
                (x, y, visibility)
        output_dir: Path to save visualization outputs.
            Defaults to "predictions_visualizations"
        num_samples: Number of images to visualize. If None, processes all images
        random_select: If True, randomly samples images; if False, uses first N images
        show_ground_truth: If True, displays ground truth poses alongside predictions.
            If False, only shows predictions but uses GT visibility mask
        plot_bboxes: If True and the model is a top-down model, predicted bboxes will
            be shown in the images as well
    """
    # Setup output directory
    output_dir = Path(output_dir or "predictions_visualizations")
    output_dir.mkdir(exist_ok=True)

    # Select images to process
    image_paths = list(predictions.keys())
    if num_samples and num_samples < len(image_paths):
        if random_select:
            image_paths = np.random.choice(
                image_paths, num_samples, replace=False
            ).tolist()
        else:
            image_paths = image_paths[:num_samples]

    # Process each selected image
    for image_path in image_paths:
        # Get prediction and ground truth data
        pred_data = predictions[image_path]
        gt_keypoints = ground_truth[image_path]  # Shape: [N, num_keypoints, 3]

        # Create visibility mask from first GT sample. This mask will be applied to all samples for consistency
        vis_mask = gt_keypoints[0, :, 2] > 0

        # Process ground truth keypoints if showing GT
        if show_ground_truth:
            visible_gt = []
            for gt in gt_keypoints:
                visible_points = gt[vis_mask, :2]  # Keep only x,y for visible joints
                visible_gt.append(visible_points)
            visible_gt = np.stack(visible_gt)  # Shape: [N, num_visible_joints, 2]
        else:
            visible_gt = None

        # Process predicted keypoints
        pred_keypoints = pred_data["bodyparts"]  # Shape: [N, num_keypoints, 3]
        visible_pred = []
        for pred in pred_keypoints:
            visible_points = pred[vis_mask]  # Keep only visible joint predictions
            visible_pred.append(visible_points)
        visible_pred = np.stack(visible_pred)  # Shape: [N, num_visible_joints, 3]

        if plot_bboxes:
            bboxes = predictions[image_path].get("bboxes", None)
            bbox_scores = predictions[image_path].get("bbox_scores", None)
            bounding_boxes = (
                (bboxes, bbox_scores)
                if bboxes is not None and bbox_scores is not None
                else None
            )
        else:
            bounding_boxes = None

        # Generate and save visualization
        try:
            plot_gt_and_predictions(
                image_path=image_path,
                output_dir=output_dir,
                gt_bodyparts=visible_gt,
                pred_bodyparts=visible_pred,
                bounding_boxes=bounding_boxes,
            )
            print(f"Successfully plotted predictions for {image_path}")
        except Exception as e:
            print(f"Error plotting predictions for {image_path}: {str(e)}")


def plot_gt_and_predictions(
    image_path: str | Path,
    output_dir: str | Path,
    gt_bodyparts: np.ndarray,
    pred_bodyparts: np.ndarray,
    gt_unique_bodyparts: np.ndarray | None = None,
    pred_unique_bodyparts: np.ndarray | None = None,
    mode: str = "bodypart",
    colormap: str = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.7,
    p_cutoff: float | list[float] = 0.6,
    bounding_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    bboxes_pcutoff: float = 0.6,
    bounding_boxes_color: str = "auto",
):
    """Plot ground truth and predictions on an image.

    Args:
        image_path: Path to the image
        gt_bodyparts: Ground truth keypoints array (num_animals, num_keypoints, 3)
        pred_bodyparts: Predicted keypoints array (num_animals, num_keypoints, 3)
        output_dir: Directory where labeled images will be saved
        gt_unique_bodyparts: Ground truth unique bodyparts if any
        pred_unique_bodyparts: Predicted unique bodyparts if any
        mode: How to color the points ("bodypart" or "individual")
        colormap: Matplotlib colormap name
        dot_size: Size of the plotted points
        alpha_value: Transparency of the points
        p_cutoff: Confidence threshold for showing predictions. If a list is provided,
            there should be one value for each bodypart and one value for each unique
            bodypart (if there are any).
        bounding_boxes:  bounding boxes (top-left corner, size) and their respective
            confidence levels,
        bboxes_pcutoff: bounding boxes confidence cutoff threshold.
        bounding_boxes_color: If plotting bounding boxes, this is the color that will be
            used for bounding boxes. If set to "auto" (default value):
                - if mode is "bodypart", the bbox color will be a default color
                - if mode is "individual", each individual's color will be used for its
                    bounding box
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the image
    frame = auxfun_videos.imread(str(image_path), mode="skimage")
    num_pred, num_keypoints = pred_bodyparts.shape[:2]

    # Create figure and set dimensions
    fig, ax = create_minimal_figure()
    h, w, _ = np.shape(frame)
    fig.set_size_inches(w / 100, h / 100)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.imshow(frame, "gray")

    # Set up colors based on mode
    if mode == "bodypart":
        num_colors = num_keypoints
        if pred_unique_bodyparts is not None:
            num_colors += pred_unique_bodyparts.shape[1]
        colors = get_cmap(num_colors, name=colormap)

        predictions = pred_bodyparts.swapaxes(0, 1)
        ground_truth = gt_bodyparts.swapaxes(0, 1)
    elif mode == "individual":
        colors = get_cmap(num_pred + 1, name=colormap)
        predictions = pred_bodyparts
        ground_truth = gt_bodyparts
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if bounding_boxes_color == "auto":
        if mode == "bodypart":
            bboxes_color = None
        elif mode == "individual":
            bboxes_color = get_cmap(num_pred + 1, name=colormap)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    else:
        bboxes_color = bounding_boxes_color

    # Plot regular bodyparts
    ax = make_multianimal_labeled_image(
        frame,
        ground_truth,
        predictions[:, :, :2],
        predictions[:, :, 2:],
        colors,
        dot_size,
        alpha_value,
        p_cutoff,
        ax=ax,
        bounding_boxes=bounding_boxes,
        bboxes_cutoff=bboxes_pcutoff,
        bboxes_color=bboxes_color,
    )

    # Plot unique bodyparts if present
    if pred_unique_bodyparts is not None and gt_unique_bodyparts is not None:
        if mode == "bodypart":
            unique_predictions = pred_unique_bodyparts.swapaxes(0, 1)
            unique_ground_truth = gt_unique_bodyparts.swapaxes(0, 1)
        else:
            unique_predictions = pred_unique_bodyparts
            unique_ground_truth = gt_unique_bodyparts

        ax = make_multianimal_labeled_image(
            frame,
            unique_ground_truth,
            unique_predictions[:, :, :2],
            unique_predictions[:, :, 2:],
            colors[num_keypoints:],
            dot_size,
            alpha_value,
            p_cutoff,
            ax=ax,
        )

    # Save the labeled image
    save_labeled_frame(
        fig,
        str(image_path),
        str(output_dir),
        belongs_to_train=False,
    )
    erase_artists(ax)
    plt.close()


def evaluate_snapshot(
    cfg: dict,
    loader: DLCLoader,
    snapshot: Snapshot,
    scorer: str,
    transform: A.Compose | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    detector_snapshot: Snapshot | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Evaluates a snapshot.
    The evaluation results are stored in the .h5 and .csv file under the subdirectory
    'evaluation_results'.

    Args:
        cfg: the content of the project's config file
        loader: the loader for the shuffle to evaluate
        snapshot: the snapshot to evaluate
        scorer: the scorer name to use for the snapshot
        transform: transformation pipeline for evaluation
            ** Should normalise the data the same way it was normalised during training **
        plotting: Plots the predictions on the train and test images. If provided it must
            be either ``True``, ``False``, ``"bodypart"``, or ``"individual"``. Setting
            to ``True`` defaults as ``"bodypart"`` for multi-animal projects.
        show_errors: whether to compare predictions and ground truth
        comparison_bodyparts: A subset of the bodyparts for which to compute the
            evaluation metrics.
        per_keypoint_evaluation: Compute the train and test RMSE for each keypoint, and
            save the results to a {model_name}-keypoint-results.csv in the
            evaluation-results-pytorch folder.
        detector_snapshot: Only for TD models. If defined, evaluation metrics are
            computed using the detections made by this snapshot
        pcutoff: The cutoff to use for computing evaluation metrics. When `None`, the
            cutoff will be loaded from the project config. If a list is provided, there
            should be one value for each bodypart and one value for each unique bodypart
            (if there are any). If a dict is provided, the keys should be bodyparts
            mapping to pcutoff values for each bodypart. Bodyparts that are not defined
            in the dict will have pcutoff set to 0.6.
    """
    head_type = loader.model_cfg["model"]["heads"]["bodypart"]["type"]
    if head_type == "DLCRNetHead":
        prune_paf_graph.benchmark_paf_graphs(
            loader=loader,
            snapshot_path=snapshot.path,
            verbose=False,
        )

    parameters = loader.get_dataset_parameters()

    detector_path = None
    if detector_snapshot is not None:
        detector_path = detector_snapshot.path

    pose_runner, detector_runner = get_inference_runners(
        model_config=loader.model_cfg,
        snapshot_path=snapshot.path,
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=loader.model_cfg["metadata"]["with_identity"],
        transform=transform,
        detector_path=detector_path,
    )

    # For memory-replay SuperAnimal models, convert bodyparts to project bodyparts
    if weight_init_cfg := loader.model_cfg["train_settings"].get("weight_init", None):
        weight_init = WeightInitialization.from_dict(weight_init_cfg)
        if weight_init.memory_replay:
            bodyparts = weight_init.bodyparts
            if bodyparts is None:
                bodyparts = auxiliaryfunctions.get_bodyparts(cfg)

            parameters = PoseDatasetParameters(
                bodyparts=bodyparts,
                unique_bpts=parameters.unique_bpts,
                individuals=parameters.individuals,
            )

    # get the names of bodyparts on which the model is evaluated
    eval_parameters = PoseDatasetParameters(
        bodyparts=_get_subset_bodyparts(parameters.bodyparts, comparison_bodyparts),
        unique_bpts=_get_subset_bodyparts(parameters.unique_bpts, comparison_bodyparts),
        individuals=parameters.individuals,
    )

    if pcutoff is None:
        pcutoff = cfg.get("pcutoff", 0.6)
    elif isinstance(pcutoff, dict):
        pcutoff = [
            pcutoff.get(bpt, 0.6)
            for bpt in eval_parameters.bodyparts + eval_parameters.unique_bpts
        ]
    _validate_pcutoff(parameters.bodyparts, parameters.unique_bpts, pcutoff)

    predictions = {}
    rmse_per_bodypart = {}
    bounding_boxes = {}
    scores = {
        "%Training dataset": loader.train_fraction,
        "Shuffle number": loader.shuffle,
        "Training epochs": snapshot.epochs,
        "Detector epochs (TD only)": (
            -1 if detector_snapshot is None else detector_snapshot.epochs
        ),
        "pcutoff": (
            ", ".join([str(v) for v in pcutoff])
            if isinstance(pcutoff, list)
            else pcutoff
        ),
    }
    for split in ["train", "test"]:
        results, predictions_for_split = evaluate(
            pose_runner=pose_runner,
            loader=loader,
            mode=split,
            pcutoff=pcutoff,
            detector_runner=detector_runner,
            comparison_bodyparts=comparison_bodyparts,
            per_keypoint_evaluation=per_keypoint_evaluation,
            parameters=parameters,
        )
        if per_keypoint_evaluation:
            rmse_per_bodypart[split] = _extract_rmse_per_bodypart(
                results,
                eval_parameters.bodyparts,
                eval_parameters.unique_bpts,
            )

        df_split_predictions = build_predictions_dataframe(
            scorer=scorer,
            predictions=predictions_for_split,
            parameters=eval_parameters,
            image_name_to_index=image_to_dlc_df_index,
        )
        split_bounding_boxes = build_bboxes_dict_for_dataframe(
            predictions=predictions_for_split,
            image_name_to_index=image_to_dlc_df_index,
        )
        predictions[split] = df_split_predictions
        bounding_boxes[split] = split_bounding_boxes
        for k, v in results.items():
            scores[f"{split} {k}"] = round(v, 2)

    results_filename = f"{scorer}.h5"
    df_predictions = pd.concat(predictions.values(), axis=0)
    df_predictions = df_predictions.reindex(loader.df.index)
    output_filename = loader.evaluation_folder / results_filename
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_hdf(output_filename, key="df_with_missing")

    df_scores = pd.DataFrame([scores]).set_index(
        [
            "%Training dataset",
            "Shuffle number",
            "Training epochs",
            "Detector epochs (TD only)",
            "pcutoff",
        ]
    )
    scores_filepath = output_filename.with_suffix(".csv")
    scores_filepath = scores_filepath.with_stem(scores_filepath.stem + "-results")
    print(f"Evaluation results file: {scores_filepath.name}")
    save_evaluation_results(df_scores, scores_filepath, show_errors, pcutoff)

    if per_keypoint_evaluation:
        rmse_per_bpt_path = output_filename.with_name(
            output_filename.stem + "-keypoint-results.csv"
        )
        save_rmse_per_bodypart(rmse_per_bodypart, rmse_per_bpt_path, show_errors)

    if plotting:
        folder_name = f"LabeledImages_{scorer}"
        folder_path = loader.evaluation_folder / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        if isinstance(plotting, str):
            plot_mode = plotting
        else:
            plot_mode = "bodypart"

        df_ground_truth = ensure_multianimal_df_format(loader.df)

        bboxes_cutoff = (
            loader.model_cfg.get("detector", {})
            .get("model", {})
            .get("box_score_thresh", 0.6)
        )

        for mode in ["train", "test"]:
            df_combined = predictions[mode].merge(
                df_ground_truth, left_index=True, right_index=True
            )
            bboxes_split = bounding_boxes[mode]

            plot_evaluation_results(
                df_combined=df_combined,
                project_root=cfg["project_path"],
                scorer=cfg["scorer"],
                model_name=scorer,
                output_folder=str(folder_path),
                in_train_set=mode == "train",
                plot_unique_bodyparts=eval_parameters.num_unique_bpts > 0,
                mode=plot_mode,
                colormap=cfg["colormap"],
                dot_size=cfg["dotsize"],
                alpha_value=cfg["alphavalue"],
                p_cutoff=cfg["pcutoff"],
                bounding_boxes=bboxes_split,
                bboxes_cutoff=bboxes_cutoff,
            )

    return df_predictions


def evaluate_network(
    config: str | Path,
    shuffles: Iterable[int] = (1,),
    trainingsetindex: int | str = 0,
    snapshotindex: int | str | None = None,
    device: str | None = None,
    plotting: bool | str = False,
    show_errors: bool = True,
    transform: A.Compose = None,
    snapshots_to_evaluate: list[str] | None = None,
    comparison_bodyparts: str | list[str] | None = None,
    per_keypoint_evaluation: bool = False,
    modelprefix: str = "",
    detector_snapshot_index: int | None = None,
    pcutoff: float | list[float] | dict[str, float] | None = None,
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
        snapshots_to_evaluate: List of snapshot names to evaluate (e.g. ["snapshot-50",
            "snapshot-75"]). If defined, `snapshotindex` will be ignored.
        comparison_bodyparts: A subset of the bodyparts for which to compute the
            evaluation metrics.
        per_keypoint_evaluation: Compute the train and test RMSE for each keypoint, and
            save the results to a {model_name}-keypoint-results.csv in the
            evaluation-results-pytorch folder.
        modelprefix: directory containing the deeplabcut models to use when evaluating
            the network. By default, they are assumed to exist in the project folder.
        detector_snapshot_index: Only for TD models. If defined, uses the detector with
            the given index for pose estimation.
        pcutoff: The cutoff to use for computing evaluation metrics. When `None`, the
            cutoff will be loaded from the project config. If a list is provided, there
            should be one value for each bodypart and one value for each unique bodypart
            (if there are any). If a dict is provided, the keys should be bodyparts
            mapping to pcutoff values for each bodypart. Bodyparts that are not defined
            in the dict will have pcutoff set to 0.6.

    Examples:
        If you want to evaluate on shuffle 1 without plotting predictions.

        >>> import deeplabcut
        >>> deeplabcut.evaluate_network(
        >>>     '/analysis/project/reaching-task/config.yaml', shuffles=[1],
        >>> )

        If you want to evaluate shuffles 0 and 1 and plot the predictions.

        >>> deeplabcut.evaluate_network(
        >>>     '/analysis/project/reaching-task/config.yaml',
        >>>     shuffles=[0, 1],
        >>>     plotting=True,
        >>> )

        If you want to plot assemblies for a maDLC project

        >>> deeplabcut.evaluate_network(
        >>>     '/analysis/project/reaching-task/config.yaml',
        >>>     shuffles=[1],
        >>>     plotting="individual",
        >>> )
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

    if detector_snapshot_index is None:
        detector_snapshot_index = cfg["detector_snapshotindex"]

    for train_set_index in train_set_indices:
        for shuffle in shuffles:
            loader = DLCLoader(
                config=config,
                shuffle=shuffle,
                trainset_index=train_set_index,
                modelprefix=modelprefix,
            )
            loader.evaluation_folder.mkdir(exist_ok=True, parents=True)

            if device is not None:
                loader.model_cfg["device"] = device
            loader.model_cfg["device"] = utils.resolve_device(loader.model_cfg)

            snapshots = get_model_snapshots(
                snapshotindex,
                model_folder=loader.model_folder,
                task=loader.pose_task,
                snapshot_filter=snapshots_to_evaluate,
            )

            detector_snapshots = [None]
            if loader.pose_task == Task.TOP_DOWN:
                if detector_snapshot_index is not None:
                    det_snapshots = get_model_snapshots(
                        "all", loader.model_folder, Task.DETECT
                    )
                    if len(det_snapshots) == 0:
                        print(
                            "The detector_snapshot_index was set to "
                            f"{detector_snapshot_index} but no detector snapshots were "
                            f"found in {loader.model_folder}. Using ground truth "
                            "bounding boxes to compute metrics.\n"
                            "To analyze videos with a top-down model, you'll need to "
                            "train a detector!"
                        )
                    else:
                        detector_snapshots = get_model_snapshots(
                            detector_snapshot_index,
                            loader.model_folder,
                            Task.DETECT,
                        )
                else:
                    print("Using GT bounding boxes to compute evaluation metrics")

            for detector_snapshot in detector_snapshots:
                for snapshot in snapshots:
                    scorer = get_scorer_name(
                        cfg=cfg,
                        shuffle=shuffle,
                        train_fraction=loader.train_fraction,
                        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
                        modelprefix=modelprefix,
                    )
                    print(f"Evaluation scorer: {scorer}")
                    evaluate_snapshot(
                        loader=loader,
                        cfg=cfg,
                        scorer=scorer,
                        snapshot=snapshot,
                        transform=transform,
                        plotting=plotting,
                        show_errors=show_errors,
                        comparison_bodyparts=comparison_bodyparts,
                        per_keypoint_evaluation=per_keypoint_evaluation,
                        detector_snapshot=detector_snapshot,
                        pcutoff=pcutoff,
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
    df_scores: pd.DataFrame, scores_path: Path, print_results: bool, pcutoff: float
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
        df_existing_results = pd.read_csv(
            combined_scores_path, index_col=[0, 1, 2, 3, 4]
        )
        df_scores = df_scores.combine_first(df_existing_results)

    df_scores = df_scores.sort_index()
    df_scores.to_csv(combined_scores_path)


def save_rmse_per_bodypart(
    rmse_per_bodypart: dict[str, dict[str, float]],
    output_path: Path,
    print_results: bool,
) -> None:
    """
    Saves the evaluation results per bodypart to a CSV file.

    Args:
        rmse_per_bodypart: The scores dataframe for a snapshot
        output_path: The path of the file where
        print_results: Whether to print results to the console
    """
    index, data = [], []
    if print_results:
        print(f"Per-bodypart evaluation results ({output_path.stem}):")

    for split, rmse_results in rmse_per_bodypart.items():
        key = split.capitalize() + " error (px)"
        index.append(key)
        data.append(rmse_results)

        if print_results:
            print(f"  {key}")
            bpt_key_length = max([len(k) for k in rmse_results.keys()]) + 4
            for k, v in rmse_results.items():
                key = (k + ":").ljust(bpt_key_length)
                print(f"    {key}{v:3>.2f}px")

    # Save scores file
    df_rmse_per_bodypart = pd.DataFrame(data, index=index)
    df_rmse_per_bodypart.to_csv(output_path)


def _validate_pcutoff(
    bodyparts: list[str],
    unique_bpts: list[str],
    pcutoff: float | list[float],
) -> None:
    """Checks that the given `pcutoff` value has the correct number of elements"""
    if isinstance(pcutoff, (int, float)):
        return

    total_bodyparts = len(bodyparts) + len(unique_bpts)
    if len(pcutoff) != total_bodyparts:
        raise ValueError(
            "When passing the pcutoff as a list, the length of the list should be "
            "equal to the number of bodyparts and the number of unique bpts. "
            f"Found a list containing {len(pcutoff)} elements, but there are "
            f"{total_bodyparts} total bodyparts, which are {bodyparts + unique_bpts}."
        )


def _get_keypoints_to_use(
    bodyparts: list[str],
    bodypart_subset: str | list[str] | None,
) -> list[int] | None:
    """Computes the indices of the keypoints indices to keep based on the given subset.

    Args:
        bodyparts: The bodyparts predicted by the model.
        bodypart_subset: The subset of bodyparts to keep. If None or "all", all
            bodyparts are kept.

    Returns:
        None if all bodyparts should be kept, or bodyparts is an empty list. Otherwise,
        returns a list containing the indices of the bodyparts to keep. If no bodyparts
        should be kept, returns an empty list.
    """
    if len(bodyparts) == 0 or bodypart_subset is None or bodypart_subset == "all":
        return None

    if isinstance(bodypart_subset, str):
        bodypart_subset = [bodypart_subset]

    to_keep = set(bodypart_subset)
    return [i for i, b in enumerate(bodyparts) if b in to_keep]


def _get_subset_bodyparts(
    bodyparts: list[str],
    subset: str | list[str] | None,
) -> list[str]:
    """Gets a subset of bodyparts that were used.

    Args:
        bodyparts: The bodyparts output by the model.
        subset: The subset of bodyparts to keep.

    Returns:
        The bodyparts that were used to evaluate the model.
    """
    if subset is None or subset == "all":
        return bodyparts

    if isinstance(subset, str):
        subset = [subset]

    to_keep = set(subset)
    return [b for b in bodyparts if b in to_keep]


def _extract_rmse_per_bodypart(
    results: dict[str, float],
    bodyparts: list[str],
    unique_bodyparts: list[str],
) -> dict[str, float]:
    """Extracts the RMSE per bodypart metrics from the results dict

    This method modifies the given dict in-place, removing all keys for RMSE per
    bodypart or unique bodypart.

    Args:
        results: The results returned by the evaluation method.
        bodyparts: The bodyparts defined for the project.
        unique_bodyparts: The unique bodyparts defined for the project.

    Returns:
        The per-bodypart RMSE.
    """
    rmse_per_bodypart = {}
    for bpt_idx, bpt in enumerate(bodyparts):
        rmse = results.pop(f"rmse_keypoint_{bpt_idx}", None)
        if rmse is not None:
            rmse_per_bodypart[bpt] = rmse

    for bpt_idx, bpt in enumerate(unique_bodyparts):
        rmse = results.pop(f"rmse_unique_keypoint_{bpt_idx}", None)
        if rmse is not None:
            rmse_per_bodypart[bpt] = rmse

    return rmse_per_bodypart


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
