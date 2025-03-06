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
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.patches as patches

def predict(
    pose_task: Task,
    pose_runner: InferenceRunner,
    loader: Loader,
    mode: str,
    detector_runner: InferenceRunner | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Predicts poses on data contained in a loader

    Args:
        pose_task: Whether the model is a top-down or bottom-up model
        pose_runner: The runner to use for pose estimation
        loader: The loader containing the data to predict poses on
        mode: {"train", "test"} The mode to predict on
        detector_runner: If the task is "TD", a detector runner can be given to detect
            individuals in the images. If no detector is given, ground truth bounding
            boxes will be used to crop individuals before pose estimation

    Returns:
        The paths of images for which predictions were computed mapping to the
        different predictions made by each model head
    """
    image_paths = loader.image_filenames(mode)
    context = None

    if pose_task == Task.TOP_DOWN:
        # Get bounding boxes for context
        if detector_runner is not None:
            bbox_predictions = detector_runner.inference(images=tqdm(image_paths))
            context = bbox_predictions
        else:
            ground_truth_bboxes = loader.ground_truth_bboxes(mode=mode)
            context = [{"bboxes": ground_truth_bboxes[image]} for image in image_paths]

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
    pose_task: Task,
    pose_runner: InferenceRunner,
    loader: Loader,
    mode: str,
    detector_runner: InferenceRunner | None = None,
    pcutoff: float = 1,
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]]]:
    """
    Args:
        pose_task: Whether to run top-down or bottom-up
        pose_runner: The runner for pose estimation
        loader: The loader containing the data to evaluate
        mode: Either 'train' or 'test'
        detector_runner: If task == 'TD', a detector can be given to compute bounding
            boxes for pose estimation. If no detector is given, ground truth bounding
            boxes are used
        pcutoff: The p-cutoff to use for evaluation

    Returns:
        A dict containing the evaluation results
        A dict mapping the paths of images for which predictions were computed to the
            different predictions made by each model head
    """
    parameters = loader.get_dataset_parameters()
    predictions = predict(
        pose_task=pose_task,
        pose_runner=pose_runner,
        loader=loader,
        mode=mode,
        detector_runner=detector_runner,
    )
    if weight_init_cfg := loader.model_cfg["train_settings"].get("weight_init"):
        weight_init = WeightInitialization.from_dict(weight_init_cfg)
        if weight_init.memory_replay:
            for _, pred in predictions.items():
                pred["bodyparts"] = pred["bodyparts"][:, weight_init.conversion_array]

    poses = {filename: pred["bodyparts"] for filename, pred in predictions.items()}

    gt_keypoints = loader.ground_truth_keypoints(mode)
    unique_poses = None
    gt_unique_keypoints = None
    if parameters.num_unique_bpts > 1:
        unique_poses = {
            filename: pred["unique_bodyparts"] for filename, pred in predictions.items()
        }
        gt_unique_keypoints = loader.ground_truth_keypoints(mode, unique_bodypart=True)

    results = metrics.compute_metrics(
        gt_keypoints,
        poses,
        single_animal=parameters.max_num_animals == 1,
        pcutoff=pcutoff,
        unique_bodypart_poses=unique_poses,
        unique_bodypart_gt=gt_unique_keypoints,
    )

    if loader.model_cfg["metadata"]["with_identity"]:
        pred_id_scores = {
            filename: pred["identity_scores"] for filename, pred in predictions.items()
        }
        id_scores = metrics.compute_identity_scores(
            individuals=parameters.individuals,
            bodyparts=parameters.bodyparts,
            predictions=poses,
            identity_scores=pred_id_scores,
            ground_truth=gt_keypoints,
        )
        for name, score in id_scores.items():
            results[f"id_head_{name}"] = score

    # Updating poses to be aligned and padded
    for image, pose in poses.items():
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
    skeleton: list | None = None,
) -> None:
    """Visualize model predictions alongside ground truth keypoints.

    This function processes keypoint predictions and ground truth data, applies visibility
    masks, and generates visualization plots. It supports random or sequential sampling
    of images for visualization.

    Args:
        predictions: Dictionary mapping image paths to prediction data.
            Each prediction contains:
            - bodyparts: array of shape [N, num_keypoints, 3] where 3 represents (x, y, confidence)
            - bboxes: array of shape [N, 4] for bounding boxes (optional)
            - bbox_scores: array of shape [N,] for bbox confidences (optional)

        ground_truth: Dictionary mapping image paths to ground truth keypoints.
            Each value has shape [N, num_keypoints, 3] where 3 represents (x, y, visibility)

        output_dir: Path to save visualization outputs.
            Defaults to "predictions_visualizations"

        num_samples: Number of images to visualize. If None, processes all images

        random_select: If True, randomly samples images; if False, uses first N images

        show_ground_truth: If True, displays ground truth poses alongside predictions.
                          If False, only shows predictions but uses GT visibility mask
                          
        plot_bboxes: Whether to plot bounding boxes if available

        skeleton: List of joint pairs defining the skeleton connections.
                 Each pair should be a tuple of indices corresponding to the joints to connect.
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
        # Read image to get dimensions
        frame = auxfun_videos.imread(str(image_path), mode="skimage")
        h, w = frame.shape[:2]
        
        # Calculate adaptive dot size based on image dimensions
        # This creates dots that scale with image size while staying reasonable
        dot_size = int(min(w, h) * 0.015)  # 1.5% of smallest dimension
        dot_size = max(6, min(dot_size, 25))  # Keep size between 6 and 25 pixels

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
                if bbox_scores is not None and bbox_scores is not None
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
                dot_size=dot_size,
                skeleton=skeleton,
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
    p_cutoff: float = 0.6,
    bounding_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    bounding_boxes_color="k",
    bboxes_pcutoff: float = 0.6,
    skeleton: list | None = None,
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
        p_cutoff: Confidence threshold for showing predictions
        bounding_boxes:  bounding boxes (top-left corner, size) and their respective confidence levels,
        bounding_boxes_color: If bounding_boxes is not None, this is the color that will be used for plotting them
        bboxes_cutoff: bounding boxes confidence cutoff threshold.
        skeleton: List of joint pairs defining the skeleton connections.
                 Each pair should be a tuple of indices corresponding to the joints to connect.
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
        bounding_boxes_color=bounding_boxes_color,
        bboxes_cutoff=bboxes_pcutoff,
        skeleton=skeleton,
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
    dot_size = int(max(4, min(dot_size, 15)))*0.8  # Tighter bounds for dots
    
    # filter out the individuals that without GT keypoints 
    if bounding_boxes is not None:
        # filter out the individuals that without GT keypoints 
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
        
        # print(f"valid_individuals: {valid_individuals}")
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

    # Track existing text positions to avoid overlap
    existing_text_positions = []
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
                            
                            while any(abs(x_text - ex) < 50 * scale_factor and abs(y_text - ey) < 20 * scale_factor 
                                    for ex, ey in existing_text_positions):
                                y_text += 20 * scale_factor
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
                        
                        # Avoid overlapping with existing text
                        while any(abs(x_text - ex) < 50 * scale_factor and abs(y_text - ey) < 20 * scale_factor 
                                for ex, ey in existing_text_positions):
                            y_text += 20 * scale_factor
                            if y_text > h - 10:  # If we run out of vertical space
                                y_text = pred_bodyparts[idx_individual, idx_keypoint, 1]  # Reset to original y
                                x_text += 50 * scale_factor  # Move text horizontally instead
                        
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
                        for [idx1, idx2] in skeleton:
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
                            
                            # if center_hip (26) is below the p_cutoff, and root_tail (33) is above the p_cutoff,
                            # then we can use root_tail to replace center_hip (just for connection!), otherwise we use center_hip
                            # if idx1 == 26 and pred_bodyparts[idx_individual, 26, 2] < p_cutoff and pred_bodyparts[idx_individual, 33, 2] > p_cutoff:
                            #     # Replace center_hip with root_tail for this connection
                            #     if pred_bodyparts[idx_individual, idx2, 2] > p_cutoff:
                            #         connection_pairs.append({
                            #             'start': (pred_bodyparts[idx_individual, 33, 0], 
                            #                     pred_bodyparts[idx_individual, 33, 1]),
                            #             'end': (pred_bodyparts[idx_individual, idx2, 0], 
                            #                 pred_bodyparts[idx_individual, idx2, 1])
                            #         })
                            # elif idx2 == 26 and pred_bodyparts[idx_individual, 26, 2] < p_cutoff and pred_bodyparts[idx_individual, 33, 2] > p_cutoff:
                            #     # Handle case where center_hip is the end point
                            #     if pred_bodyparts[idx_individual, idx1, 2] > p_cutoff:
                            #         connection_pairs.append({
                            #             'start': (pred_bodyparts[idx_individual, idx1, 0], 
                            #                     pred_bodyparts[idx_individual, idx1, 1]),
                            #             'end': (pred_bodyparts[idx_individual, 33, 0], 
                            #                 pred_bodyparts[idx_individual, 33, 1])
                            #         })
                                    
                            # if left hip (idx: 24) is below the p_cutoff and left knee (idx: 27) is above the p_cutoff,
                            # if center hip (idx: 26) is above the p_cutoff, then connect left knee to center hip,
                            # if center hip (idx: 26) is below the p_cutoff and root_tail (idx: 33) is above the p_cutoff, then we connect left knee to root_tail
                            if pred_bodyparts[idx_individual, 24, 2] < p_cutoff and pred_bodyparts[idx_individual, 26, 2] > p_cutoff and pred_bodyparts[idx_individual, 27, 2] > p_cutoff:
                                connection_pairs.append({
                                    'start': (pred_bodyparts[idx_individual, 27, 0], 
                                            pred_bodyparts[idx_individual, 27, 1]),
                                    'end': (pred_bodyparts[idx_individual, 26, 0], 
                                        pred_bodyparts[idx_individual, 26, 1])
                                })
                                
                            # if right hip (idx: 25) is below the p_cutoff, and center hip (idx: 26) and right knee (idx: 28) are above the p_cutoff,
                            # then we can draw a line from the right knee (idx: 28) to the center hip (idx: 26)
                            if pred_bodyparts[idx_individual, 25, 2] < p_cutoff and pred_bodyparts[idx_individual, 26, 2] > p_cutoff and pred_bodyparts[idx_individual, 28, 2] > p_cutoff:
                                connection_pairs.append({
                                    'start': (pred_bodyparts[idx_individual, 28, 0], 
                                            pred_bodyparts[idx_individual, 28, 1]),
                                    'end': (pred_bodyparts[idx_individual, 26, 0], 
                                        pred_bodyparts[idx_individual, 26, 1])
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
    detector_snapshot: Snapshot | None = None,
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
        detector_snapshot: Only for TD models. If defined, evaluation metrics are
            computed using the detections made by this snapshot
    """
    pose_task = Task(loader.model_cfg.get("method", "bu"))
    parameters = loader.get_dataset_parameters()
    pcutoff = cfg.get("pcutoff", 0.6)

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
        detector_transform=None,
    )

    # The project bodyparts might be different to the bodyparts the model was trained to
    #  output, if the model is fine-tuned from SuperAnimal with memory replay.
    #  For evaluation, we want to only use the project bodyparts
    project_bodyparts = auxiliaryfunctions.get_bodyparts(cfg)
    parameters = PoseDatasetParameters(
        bodyparts=project_bodyparts,
        unique_bpts=parameters.unique_bpts,
        individuals=parameters.individuals,
    )

    predictions = {}
    bounding_boxes = {}
    scores = {
        "%Training dataset": loader.train_fraction,
        "Shuffle number": loader.shuffle,
        "Training epochs": snapshot.epochs,
        "Detector epochs (TD only)": (
            -1 if detector_snapshot is None else detector_snapshot.epochs
        ),
        "pcutoff": pcutoff,
    }
    for split in ["train", "test"]:
        results, predictions_for_split = evaluate(
            pose_task=pose_task,
            pose_runner=pose_runner,
            loader=loader,
            mode=split,
            pcutoff=pcutoff,
            detector_runner=detector_runner,
        )
        df_split_predictions = build_predictions_dataframe(
            scorer=scorer,
            predictions=predictions_for_split,
            parameters=parameters,
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
            unique_bodyparts = loader.get_dataset_parameters().unique_bpts
            bboxes_split = bounding_boxes[mode]

            plot_evaluation_results(
                df_combined=df_combined,
                project_root=cfg["project_path"],
                scorer=cfg["scorer"],
                model_name=scorer,
                output_folder=str(folder_path),
                in_train_set=mode == "train",
                plot_unique_bodyparts=len(unique_bodyparts) > 0,
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
    modelprefix: str = "",
    detector_snapshot_index: int | None = None,
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
        detector_snapshot_index: Only for TD models. If defined, uses the detector with
            the given index for pose estimation.

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

            task = Task(loader.model_cfg["method"])
            snapshots = get_model_snapshots(
                snapshotindex,
                model_folder=loader.model_folder,
                task=task,
            )

            detector_snapshots = [None]
            if task == Task.TOP_DOWN:
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
                        detector_snapshot=detector_snapshot,
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
