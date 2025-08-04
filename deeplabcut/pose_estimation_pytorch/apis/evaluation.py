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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Union, List, Tuple, Dict
import logging
import matplotlib.patches as patches

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


def visualize_predictions_PFM(
    predictions: Dict[str, Dict],
    ground_truth: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: Optional[int] = None,
    random_select: bool = False,
    plot_bboxes: bool = True,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    keypoint_vis_mask: Optional[List[int]] = None,
    keypoint_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.6
) -> None:
    """Visualize model predictions alongside ground truth keypoints with additional PFM-specific configurations."""
    # Setup output directory and logging
    output_dir = Path(output_dir or "predictions_visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure logging with a unique handler
    log_file = output_dir / "visualization.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('PFM_visualization')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f"Starting visualization process. Output directory: {output_dir}")

    # Select images to process efficiently
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

        # Process predicted keypoints
        pred_keypoints = pred_data["bodyparts"]

        if plot_bboxes:
            bboxes = predictions[image_path].get("bboxes", None)
            bbox_scores = predictions[image_path].get("bbox_scores", None)
            # this means the bboxes is the GT; so we should set the score as 1
            if bbox_scores is None:
                bbox_scores = np.ones(len(bboxes))
            # print("bboxes:", bboxes)
            # print("bbox_scores:", bbox_scores)
            bounding_boxes = (
                (bboxes, bbox_scores)
                if bbox_scores is not None and bbox_scores is not None
                else None
            )
        else:
            bounding_boxes = None
        
        # print("bounding_boxes:", bounding_boxes)
        
        # Generate visualization
        plot_gt_and_predictions_PFM(
            image_path=image_path,
            output_dir=output_dir,
            gt_bodyparts=gt_keypoints,
            pred_bodyparts=pred_keypoints,
            bounding_boxes=bounding_boxes,
            skeleton=skeleton,
            keypoint_names=keypoint_names,
            p_cutoff=confidence_threshold,
            keypoint_vis_mask=keypoint_vis_mask, # Pass the mask to plotting function
        )
        logger.info(f"Successfully visualized predictions for {image_path}")

    # Clean up logging handler
    logger.removeHandler(handler)
    handler.close()

def get_dynamic_skeleton(skeleton, keypoints, p_cutoff=0.6):
    """
    Modify skeleton connections based on keypoint confidence scores.
    
    If certain keypoints have low confidence (below threshold), alternative 
    skeleton connections will be used instead of the original ones.
    
    Args:
        skeleton (list): List of tuples/lists representing skeleton connections as (start_idx, end_idx)
        keypoints (numpy.ndarray): Array of shape (..., 3) where the last dimension contains
                                  [x, y, confidence] for each keypoint
        p_cutoff (float): Confidence threshold (0.0-1.0)
    
    Returns:
        list: Modified skeleton connections based on confidence scores
    """
    dynamic_skeleton = skeleton.copy()
    confidences = keypoints[..., 2]  # Get confidence scores
    
    # Dictionary to store special connection rules
    # dict_name_to_idx = {name: idx for idx, name in enumerate(keypoint_name_simplified)}
    dict_name_to_idx = {"L_Shoulder" : 12, "R_Shoulder" : 13, "L_Elbow": 18, "R_Elbow": 19, "neck": 11,
                        "L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23, "L_Knee": 27, "R_Knee": 28, "L_hip": 24, "R_hip": 25, "C_hip": 26}
    
    
    # Template for special connections with rules for alternative connections
    special_connections = {
        # Format: (point_to_check, [(original_connections), (alternative_connection)])
        "L_Shoulder": [(("neck", "L_Shoulder"), ("L_Shoulder", "L_Elbow") ), ("neck", "L_Elbow")],  # L_S: ori_connection: {L_S to L_elbow, L_S to neck}; alt_connection: L_Elbow to neck if L_S is below threshold
        "R_Shoulder": [(("neck", "R_Shoulder"), ("R_Shoulder", "R_Elbow") ), ("neck", "R_Elbow")],  # R_S: ori_connection: {R_S to R_elbow, R_S to neck}; alt_connection: R_Elbow to neck if R_S is below threshold
        "L_Wrist": [(("L_Hand", "L_Wrist"), ("L_Wrist", "L_Elbow")), ("L_Hand", "L_Elbow")],  # L_W: ori_connection: {L_H to L_W, L_W to L_Elbow}; alt_connection: L_H to L_Elbow if L_W is below threshold
        "R_Wrist": [(("R_Hand", "R_Wrist"), ("R_Wrist", "R_Elbow")), ("R_Hand", "R_Elbow")],  # R_W: ori_connection: {R_H to R_W, R_W to R_Elbow}; alt_connection: R_H to R_Elbow if R_W is below threshold
        "L_hip": [(( "L_Knee", "L_hip"), ("L_hip", "C_hip")), ("L_Knee", "C_hip")], 
        "R_hip": [(( "R_Knee", "R_hip"), ("R_hip", "C_hip")), ("R_Knee", "C_hip")], 
    }
    # Process each keypoint in special connections
    for keypoint_name, (original_connections, alternative_connection) in special_connections.items():
        # Get the index of the keypoint
        keypoint_idx = dict_name_to_idx[keypoint_name]
        
        # Check if keypoint confidence is below threshold
        if confidences[keypoint_idx] < p_cutoff:
            # Convert named connections to index-based connections
            original_connections_idx = []
            for conn1, conn2 in original_connections:
                # Add both connections to the list
                original_connections_idx.append([dict_name_to_idx[conn1], dict_name_to_idx[conn2]])
                # Also consider reverse connection
                original_connections_idx.append([dict_name_to_idx[conn2], dict_name_to_idx[conn1]])
            
            # Convert alternative connection to index-based
            # todo: alternative_connection also could contain multiple connections
            alt_conn_idx = [dict_name_to_idx[alternative_connection[0]], dict_name_to_idx[alternative_connection[1]]]
            
            # Remove original connections from dynamic skeleton
            for conn in original_connections_idx:
                if conn in dynamic_skeleton:
                    dynamic_skeleton.remove(conn)
                
            # Add alternative connection if it's not already in the skeleton
            if alt_conn_idx not in dynamic_skeleton and [alt_conn_idx[1], alt_conn_idx[0]] not in dynamic_skeleton:
                dynamic_skeleton.append(alt_conn_idx) 
    return dynamic_skeleton


class DynamicSkeleton:
    def __init__(self,  pred_bodyparts, p_cutoff=0.6):
        self.keypoints = [
            "forehead",
            "head",
            "L_Eye",
            "R_Eye",
            "nose",
            "L_Ear",
            "R_Ear",
            "mouth_front_top",
            "mouth_front_bottom",
            "mouth_B_L",
            "mouth_B_R",
            "neck",
            "L_Shoulder",
            "R_Shoulder",
            "upper_B",
            "torso_M_B",
            "body_C",
            "lower_B",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "L_Hand",
            "R_Hand",
            "L_hip",
            "R_hip",
            "C_hip",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
            "L_foot",
            "R_foot",
            "root_tail",
            "M_tail",
            "M_end_tail",
            "end_tail"
        ]
        self.parent_mapping = {
            # body part
            'head': "neck",
            'neck': None, # root
            'L_Shoulder': 'neck',    # Left Shoulder
            'R_Shoulder': 'neck',    # Right Shoulder
            'L_Elbow': 'L_Shoulder',
            'R_Elbow': 'R_Shoulder',
            'L_Wrist': 'L_Elbow',
            'R_Wrist': 'R_Elbow',
            'L_Hand': 'L_Wrist',
            'R_Hand': 'R_Wrist',
            'C_hip': None,  # Hip connected to lower body
            'L_hip': 'C_hip',
            'R_hip': 'C_hip',
            'L_Knee': 'L_hip',
            'R_Knee': 'R_hip',
            'L_Ankle': 'L_Knee',
            'R_Ankle': 'R_Knee',
            'L_foot': 'L_Ankle',
            'R_foot': 'R_Ankle',
            'root_tail': 'C_hip',
            'M_tail': 'root_tail',
            'M_end_tail': 'M_tail',
            'end_tail': 'M_end_tail',
            # mouse part
            'L_Ear': 'L_Eye',
            'R_Ear': 'R_Eye',
            'L_Eye' : 'nose',
            'R_Eye' : 'nose',
            'nose' : None,
        }
        
        confidence_dict = {}
        for idx, keypoint in enumerate(self.keypoints):
            confidence_dict[keypoint] = pred_bodyparts[idx, 2]
        self.confidence_dict = confidence_dict
        self.p_cutoff = p_cutoff
        self.dynamic_skeleton = []
        
        # if C_hip is None, then we use root_tail to replace C_hip, and remove {'root_tail': 'C_hip'}
        if self.confidence_dict.get('C_hip') < self.p_cutoff:
            self.parent_mapping['L_hip'] = 'root_tail'
            self.parent_mapping['R_hip'] = 'root_tail'
            self.parent_mapping['M_tail'] = 'root_tail'
            self.parent_mapping['root_tail'] = None
            self.dynamic_skeleton.append(('root_tail', 'C_hip'))
    
    def change_name_to_idx_dynamic_skeleton(self, dynamic_skeleton):
        # change the dynamic skeleton index to the new index;
        dynamic_skeleton = []
        for idx, (from_node, end_node) in enumerate(self.dynamic_skeleton):
            # print((self.keypoints.index(from_node), self.keypoints.index(end_node)))
            dynamic_skeleton.append((self.keypoints.index(from_node), self.keypoints.index(end_node)))
        return dynamic_skeleton
        
        
    def find_nearest_ancester(self, node):
        current_node = self.parent_mapping.get(node)
        while current_node is not None:
            current_node_conf = self.confidence_dict[current_node]
            if current_node_conf > self.p_cutoff:
                return current_node
            else:
                current_node = self.parent_mapping.get(current_node)                
        return None
    
    def get_dynamic_skeleton(self):
        # only consider the keypoints that are in the parent_mapping
        for keypoint in self.parent_mapping.keys():
            keypoint_conf = self.confidence_dict[keypoint]
            if keypoint_conf > self.p_cutoff:
                ancester = self.find_nearest_ancester(keypoint)
                if ancester is not None:
                    self.dynamic_skeleton.append((ancester, keypoint))

        # add connection between C_hip and neck
        if self.confidence_dict.get('C_hip') > self.p_cutoff and self.confidence_dict.get('neck') > self.p_cutoff:
            self.dynamic_skeleton.append(('C_hip', 'neck'))
        # if conf[C_hip]<p_cutoff, then we use root_tail to replace C_hip
        elif self.confidence_dict.get('C_hip') < self.p_cutoff and self.confidence_dict.get('neck') > self.p_cutoff and self.confidence_dict.get('root_tail') > self.p_cutoff:
            self.dynamic_skeleton.append(('root_tail', 'neck'))
            
        return self.change_name_to_idx_dynamic_skeleton(self.dynamic_skeleton) 
    
def plot_gt_and_predictions_PFM(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    gt_bodyparts: Optional[np.ndarray] = None,
    pred_bodyparts: Optional[np.ndarray] = None,
    mode: str = "bodypart",
    colormap: str = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.8,
    p_cutoff: float = 0.6,
    bounding_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    bounding_boxes_color="k",
    bboxes_pcutoff: float = 0.6,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    keypoint_names: Optional[List[str]] = None,
    keypoint_vis_mask: Optional[List[int]] = None,
    labels: List[str] = ["+", ".", "x"],
) -> None:
    """Plot ground truth and predictions on an image.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save the visualization
        gt_bodyparts: Ground truth keypoints array [N, num_keypoints, 3] (x, y, vis_label)
        pred_bodyparts: Predicted keypoints array [N, num_keypoints, 3] (x, y, confidence)
        bounding_boxes: Tuple of (boxes, scores) for bounding box visualization
        dot_size: Size of the keypoint markers
        alpha_value: Transparency for points and lines
        p_cutoff: Confidence threshold for predictions
        mode: How to color the points ("bodypart" or "individual")
        colormap: Matplotlib colormap name
        bbox_color: Color for bounding boxes
        skeleton: List of joint pairs for skeleton visualization
        keypoint_names: List of keypoint names for labeling
        keypoint_vis_mask: List of keypoint indices to show (default: all keypoints visible)
        labels: Marker styles for [ground truth, reliable predictions, unreliable predictions]
    """
    # Set default keypoint visibility mask if not provided
    if pred_bodyparts is not None and keypoint_vis_mask is None:
        keypoint_vis_mask = [1] * pred_bodyparts.shape[1]  # All keypoints visible by default
    
    # Read image and calculate dot size
    frame = auxfun_videos.imread(str(image_path), mode="skimage")
    h, w = frame.shape[:2]
    # Calculate adaptive dot size based on image dimensions
    # Use a logarithmic scale to handle very large or small images better
    diagonal = np.sqrt(w * w + h * h)  # Image diagonal length
    base_size = np.log10(diagonal) * 3  # Logarithmic scaling
    # print("diagonal:", diagonal)
    # Fine-tune the dot size
    if diagonal > 1200:  # High resolution
        dot_size = base_size * 2.0
    elif diagonal < 800:  # Low resolution
        dot_size = base_size * 1.0
    else:  # Medium resolution
        dot_size = base_size
        
    # Ensure dot size stays within reasonable bounds
    dot_size = int(max(4, min(dot_size, 15)))*0.8 # *5 for oap # # Tighter bounds for dots
    
    # filter out the non exist individuals  
    if bounding_boxes is not None:
        valid_individuals = []
        for idx, bbox_score in enumerate(bounding_boxes[1]):
            if bbox_score > bboxes_pcutoff:
                valid_individuals.append(idx)
        
    # if gt_bodyparts is None:
    #     tmp_valid_bodyparts = pred_bodyparts
    # else:
    #     tmp_valid_bodyparts = gt_bodyparts
        
    # if tmp_valid_bodyparts is not None:
    #     valid_individuals = []
    #     for idx in range(tmp_valid_bodyparts.shape[0]):
    #         # Check if this individual has any valid keypoints
    #         # A keypoint is valid if its visibility (3rd value) is not -1
    #         has_valid_keypoints = False
            
    #         for kp_idx in range(tmp_valid_bodyparts.shape[1]):
    #             kp = tmp_valid_bodyparts[idx, kp_idx]
    #             # Check if keypoint is visible
    #             if kp[2] != -1:
    #                 has_valid_keypoints = True
    #                 break  # We found at least one valid keypoint, no need to check more
            
    #         # Include individual if they have at least one valid keypoint
    #         if has_valid_keypoints:
    #             valid_individuals.append(idx)
                
        # print(f"Found {len(valid_individuals)} valid individuals out of {gt_bodyparts.shape[0]}")
        # Filter both ground truth and predictions
        
        if valid_individuals:
            if gt_bodyparts is not None:
                gt_bodyparts = gt_bodyparts[valid_individuals]
            if pred_bodyparts is not None:
                pred_bodyparts = pred_bodyparts[valid_individuals]
            if bounding_boxes is not None:
                bounding_boxes = (
                    bounding_boxes[0][valid_individuals],
                    bounding_boxes[1][valid_individuals]
                )
    
    num_pred, num_keypoints = pred_bodyparts.shape[:2]
    
    # print("After filtering:")
    # print("num_pred, num_keypoints:", num_pred, num_keypoints)
    # if gt_bodyparts is not None:
        # print("gt_bodyparts shape:", gt_bodyparts.shape)
    
    # Create figure with optimal settings
    fig, ax = create_minimal_figure()
    fig.set_size_inches(w/100, h/100)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.imshow(frame, "gray")

    # Set up colors based on mode
    if mode == "bodypart":
        num_colors = num_keypoints
        # if pred_unique_bodyparts is not None:
        #     num_colors += pred_unique_bodyparts.shape[1]
        colors = get_cmap(num_colors, name=colormap)
        # print("colors:", colors)
    # predictions = pred_bodyparts.swapaxes(0, 1)
    # ground_truth = gt_bodyparts.swapaxes(0, 1)
    elif mode == "individual":
        colors = get_cmap(num_pred + 1, name=colormap)
        # predictions = pred_bodyparts
        # ground_truth = gt_bodyparts
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # print("bounding_boxes:", bounding_boxes)
    
    # Draw bounding boxes if provided
    if bounding_boxes is not None:
        # print(f"bounding_boxes: {bounding_boxes}")
        for bbox, bbox_score in zip(bounding_boxes[0], bounding_boxes[1]):
            bbox_origin = (bbox[0], bbox[1])
            (bbox_width, bbox_height) = (bbox[2], bbox[3])
            rect = patches.Rectangle(
                bbox_origin,
                bbox_width,
                bbox_height,
                linewidth=2,
                edgecolor=bounding_boxes_color,
                facecolor='none',
                linestyle="--" if bbox_score < bboxes_pcutoff else "-"
            )
            ax.add_patch(rect)

    scale_factor = min(w, h) / 1000  # Normalize scale factor based on image size

        
    plot_individual = False
    if plot_individual:
        # Save individual plots for each animal
        for idx_individual in range(num_pred):
            # print("plot individual:", idx_individual)
            # Create a new figure for each individual
            fig_ind, ax_ind = create_minimal_figure()
            fig_ind.set_size_inches(w/100, h/100)
            ax_ind.set_xlim(0, w)
            ax_ind.set_ylim(0, h)
            ax_ind.invert_yaxis()
            ax_ind.imshow(frame, "gray")
            
            # Draw bounding box for this individual if available
            if bounding_boxes is not None:
                bbox = bounding_boxes[0][idx_individual]
                bbox_score = bounding_boxes[1][idx_individual]
                bbox_origin = (bbox[0], bbox[1])
                (bbox_width, bbox_height) = (bbox[2], bbox[3])
                rect = patches.Rectangle(
                    bbox_origin,
                    bbox_width,
                    bbox_height,
                    linewidth=2,
                    edgecolor=bounding_boxes_color,
                    facecolor='none',
                    linestyle="--" if bbox_score < bboxes_pcutoff else "-"
                )
                ax_ind.add_patch(rect)
            
            # Reset text positions for each individual
            existing_text_positions = []
            
            # Plot keypoints for this individual
            for idx_keypoint in range(num_keypoints):
                if keypoint_vis_mask[idx_keypoint]:
                    
                    keypoint_confidence = pred_bodyparts[idx_individual, idx_keypoint, 2]
                    # print("keypoint_confidence_individual:", keypoint_confidence)
                    if keypoint_confidence > p_cutoff:
                        x_kp = pred_bodyparts[idx_individual, idx_keypoint, 0]
                        y_kp = pred_bodyparts[idx_individual, idx_keypoint, 1]
                        
                        ax_ind.plot(
                            x_kp, 
                            y_kp, 
                            labels[1] if keypoint_confidence > p_cutoff else labels[2], 
                            color=colors(idx_keypoint), 
                            alpha=alpha_value,
                            markersize=dot_size
                        )

                        if keypoint_names is not None:
                            # Calculate and adjust text position
                            x_text = x_kp - (10 * scale_factor)
                            y_text = y_kp - (15 * scale_factor)
                            x_text = min(max(0, x_text), w - 100)
                            y_text = min(max(0, y_text), h - 10)
                            
                            while any(abs(x_text - existing_x) < 50 * scale_factor and abs(y_text - existing_y) < 30 * scale_factor 
                                    for existing_x, existing_y in existing_text_positions):
                                y_text += 5 * scale_factor
                                if y_text > h - 10:
                                    y_text = y_kp
                                    x_text += 50 * scale_factor
                            
                            existing_text_positions.append((x_text, y_text))
                            
                            ax_ind.text(
                                x_text,
                                y_text,
                                keypoint_names[idx_keypoint], 
                                color=colors(idx_keypoint), 
                                alpha=alpha_value,
                                fontsize=dot_size * 0.8
                            )
                            
                        # Plot ground truth for this individual
                        if gt_bodyparts is not None:
                            if gt_bodyparts[idx_individual, idx_keypoint, 2] != -1:
                                ax_ind.plot(
                                    gt_bodyparts[idx_individual, idx_keypoint, 0], 
                                    gt_bodyparts[idx_individual, idx_keypoint, 1], 
                                    labels[0], 
                                    color=colors(idx_keypoint), 
                                    alpha=alpha_value,
                                    markersize=dot_size
                                )
            
            # Save individual plot
            if num_pred > 1:
                # Add index for multi-animal images
                output_path = Path(output_dir) / f"{Path(image_path).stem}_animal_{idx_individual}_predictions.png"
            else:
                # No index needed for single animal
                output_path = Path(output_dir) / f"{Path(image_path).stem}_predictions.png"
                
            plt.savefig(
                output_path,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False
            )
            plt.close(fig_ind)
    
    # Original combined plot
    # Track existing text positions to avoid overlap
    existing_text_positions = []
    
    for idx_individual in range(num_pred):
        for idx_keypoint in range(num_keypoints):
            if pred_bodyparts is not None and keypoint_vis_mask[idx_keypoint]:
                # if the keypoint is allowed to be shown and the prediction is reliable
                keypoint_confidence = pred_bodyparts[idx_individual, idx_keypoint, 2]
                if keypoint_confidence > p_cutoff:
                    pred_label = labels[1]
                else:
                    pred_label = labels[2]
                if keypoint_confidence > p_cutoff:
                    x_kp = pred_bodyparts[idx_individual, idx_keypoint, 0]
                    y_kp = pred_bodyparts[idx_individual, idx_keypoint, 1]
                    
                    ax.plot(
                        x_kp, 
                        y_kp, 
                        pred_label, 
                        color=colors(idx_keypoint), 
                        alpha=alpha_value,
                        markersize=dot_size
                    )

                    if keypoint_names is not None:
                        # Calculate initial text position
                        x_text = x_kp - (10 * scale_factor)
                        y_text = y_kp - (15 * scale_factor)
                        
                        # Ensure text stays within image bounds
                        x_text = min(max(0, x_text), w - 100)
                        y_text = min(max(0, y_text), h - 10)
                        
                        # ToDo
                        # dynamic text position;
                        # Avoid overlapping with existing text
                        while any(abs(x_text - existing_x) <= 15 * scale_factor and abs(y_text - existing_y) <= 15 * scale_factor 
                                for existing_x, existing_y in existing_text_positions):
                            y_text += 7.5 * scale_factor
                            x_text += 4 * scale_factor
                            # if y_text > h - 10:  # If we run out of vertical space
                            #     y_text = pred_bodyparts[idx_individual, idx_keypoint, 1]  # Reset to original y
                            #     x_text += 50 * scale_factor  # Move text horizontally instead
                        
                        # Record this position
                        existing_text_positions.append((x_text, y_text))
                        
                        ax.text(
                            x_text,
                            y_text,
                            keypoint_names[idx_keypoint], 
                            color=colors(idx_keypoint), 
                            alpha=alpha_value,
                            fontsize=dot_size * 0.5
                        )

                    # plot ground truth
                    if gt_bodyparts is not None:
                        if gt_bodyparts[idx_individual, idx_keypoint, 2] != -1:
                            ax.plot(
                                gt_bodyparts[idx_individual, idx_keypoint, 0], 
                                gt_bodyparts[idx_individual, idx_keypoint, 1], 
                                labels[0], 
                                color=colors(idx_keypoint), 
                                alpha=alpha_value,
                                markersize=dot_size*0.5
                            )
                    if skeleton is not None:
                        # Draw all valid connections
                        # plot the skeleton is the skeleton is not None
                        connection_pairs = []

                        # dynamic_skeleton = skeleton.copy()
                        # dynamic_skeleton = get_dynamic_skeleton(dynamic_skeleton, pred_bodyparts[idx_individual], p_cutoff)
                        
                        dynamic_skeleton = DynamicSkeleton(pred_bodyparts[idx_individual], p_cutoff).get_dynamic_skeleton()
                        
                        for [idx1, idx2] in dynamic_skeleton:
                            # idx1 = idx1 - 1
                            # idx2 = idx2 - 1
                            # Only add the connection if both keypoints are visible and have confidence above threshold
                            if (pred_bodyparts[idx_individual, idx1, 2] > p_cutoff and 
                                pred_bodyparts[idx_individual, idx2, 2] > p_cutoff):
                                connection_pairs.append({
                                    'start': (pred_bodyparts[idx_individual, idx1, 0], 
                                            pred_bodyparts[idx_individual, idx1, 1]),
                                    'end': (pred_bodyparts[idx_individual, idx2, 0], 
                                        pred_bodyparts[idx_individual, idx2, 1])
                                })
                            
                        for connection in connection_pairs:
                            ax.plot(
                                [connection['start'][0], connection['end'][0]],
                                [connection['start'][1], connection['end'][1]],
                                'g',  # black solid line
                                alpha=alpha_value * 0.8,  # slightly more transparent than points
                                linewidth=dot_size * 0.1  # scale line width with dot size
                            )
                            
    # Save the figure
    output_path = Path(output_dir) / f"{Path(image_path).stem}_predictions.png"
    # save_labeled_frame(fig, str(image_path), str(output_dir), belongs_to_train=False)
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches='tight',
        pad_inches=0,
        transparent=False
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
