#!/usr/bin/env python3
"""
Dedicated video inference implementation for superanimal_humanbody with torchvision detector.
This avoids modifying core functions and provides a clean, specific implementation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union
import torch
import torchvision.models.detection as detection
from PIL import Image
from tqdm import tqdm
import json
import logging
import yaml
import pandas as pd

from deeplabcut.pose_estimation_pytorch.apis.videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.inference import _video_inference_superanimal
from deeplabcut.modelzoo.utils import get_super_animal_scorer, get_superanimal_colormaps


def torchvision_detector_inference(images, threshold=0.1, device="cpu"):
    """
    Run the exact torchvision detector on a list of images.
    This is the working implementation that matches the Colab code.
    
    Args:
        images: list of np.ndarray or PIL.Image
        threshold: float, detection threshold
        device: str, device to run on
    Returns:
        list of dicts with 'bboxes', 'scores', and 'labels'
    """
    weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, box_score_thresh=threshold
    )
    detector.eval()
    detector.to(device)
    preprocess = weights.transforms()

    results = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        batch = [preprocess(image).to(device)]
        with torch.no_grad():
            predictions = detector(batch)[0]
        
        bboxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        # Filter for humans (COCO class 1)
        human_mask = labels == 1
        human_bboxes = bboxes[human_mask]
        human_scores = scores[human_mask]
        human_labels = labels[human_mask]
        
        # Convert to xywh format
        if len(human_bboxes) > 0:
            human_bboxes[:, 2] -= human_bboxes[:, 0]  # width = x2 - x1
            human_bboxes[:, 3] -= human_bboxes[:, 1]  # height = y2 - y1
        
        results.append({
            "bboxes": human_bboxes,
            "scores": human_scores,
            "labels": human_labels
        })
    
    return results


def video_inference_superanimal_humanbody(
    video: Union[str, Path, VideoIterator],
    model_config: dict,
    model_snapshot_path: Union[str, Path],
    detector_snapshot_path: Union[str, Path] = None,
    max_individuals: int = 1,
    bbox_threshold: float = 0.1,
    device: str = "cpu",
    cropping: List[int] = None,
    dest_folder: str = None,
    output_suffix: str = "",
) -> List[Dict[str, np.ndarray]]:
    """
    Dedicated video inference for superanimal_humanbody with torchvision detector.
    
    This implementation:
    1. Uses the exact torchvision detector
    2. Preserves detection scores and labels
    3. Handles missing detections gracefully
    4. Doesn't modify core DeepLabCut functions
    
    Args:
        video: Video path or VideoIterator
        model_config: Model configuration
        model_snapshot_path: Path to pose model snapshot
        detector_snapshot_path: Path to detector snapshot (not used, we use Colab-style detector)
        max_individuals: Maximum number of individuals to detect
        bbox_threshold: Detection threshold
        device: Device to run on
        cropping: Video cropping parameters
        dest_folder: Output folder
        output_suffix: Output file suffix
        
    Returns:
        List of predictions for each frame
    """
    
    # Initialize video iterator
    if not isinstance(video, VideoIterator):
        video = VideoIterator(str(video), cropping=cropping)
    elif cropping is not None:
        video.set_crop(cropping)

    n_frames = video.get_n_frames(robust=False)
    vid_w, vid_h = video.dimensions
    
    print(f"Starting superanimal_humanbody analysis of {video.video_path}")
    print(
        f"Video metadata: \n"
        f"  Overall # of frames:    {n_frames}\n"
        f"  Duration of video [s]:  {n_frames / max(1, video.fps):.2f}\n"
        f"  fps:                    {video.fps}\n"
        f"  resolution:             w={vid_w}, h={vid_h}\n"
    )

    # Step 1: Run Colab-style torchvision detector
    print(f"Using torchvision detector with threshold {bbox_threshold}")
    
    detector_progress = tqdm(video, desc="Detector")
    bbox_predictions = []
    
    for i, frame in enumerate(detector_progress):
        result = torchvision_detector_inference(
            images=[frame], 
            threshold=bbox_threshold, 
            device=device
        )
        bbox_predictions.extend(result)
    
    # Handle missing detections by padding with full-frame bboxes
    if len(bbox_predictions) < n_frames:
        print(f"Detector returned {len(bbox_predictions)} predictions for {n_frames} frames. Padding with full-frame bboxes.")
        for _ in range(n_frames - len(bbox_predictions)):
            bbox_predictions.append({
                'bboxes': np.array([[0, 0, vid_w, vid_h]]),
                'scores': np.array([0.0]),
                'labels': np.array([1])
            })
    elif len(bbox_predictions) > n_frames:
        print(f"Detector returned more predictions than frames. Truncating to {n_frames}.")
        bbox_predictions = bbox_predictions[:n_frames]
    
    # Rename scores to bbox_scores to match DeepLabCut expectations
    for pred in bbox_predictions:
        if 'scores' in pred:
            pred['bbox_scores'] = pred.pop('scores')
    
    video.set_context(bbox_predictions)

    # Step 2: Run pose estimation
    print(f"Running pose estimation")
    
    # Get pose inference runner
    pose_runner, _ = get_inference_runners(
        model_config=model_config,
        snapshot_path=model_snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(model_config["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_config["metadata"]["unique_bodyparts"]),
        device=device,
        detector_path=None,  # We don't use the detector runner since we already have bboxes
    )
    
    pose_progress = tqdm(video, desc="Pose")
    predictions = []
    
    for i, frame in enumerate(pose_progress):
        result = pose_runner.inference(images=[frame])
        predictions.extend(result)
    
    # Add detection context back to predictions
    for i, pred in enumerate(predictions):
        if i < len(bbox_predictions):
            pred['bboxes'] = bbox_predictions[i]['bboxes']
            pred['bbox_scores'] = bbox_predictions[i]['bbox_scores']
            if 'labels' in bbox_predictions[i]:
                pred['bbox_labels'] = bbox_predictions[i]['labels']
    
    # Log detection statistics
    frames_with_detections = sum(
        1 for pred in predictions if (
            'bboxes' in pred and len(pred['bboxes']) > 0 and 
            not np.all(pred['bboxes'] == np.array([0, 0, vid_w, vid_h]))
        )
    )
    logging.info(f"Detected individuals in {frames_with_detections} of {n_frames} frames")
    
    return predictions


def analyze_videos_superanimal_humanbody(
    config: str,
    videos: Union[str, List[str]],
    videotype: str = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    save_as_csv: bool = False,
    in_random_order: bool = False,
    snapshot_index: Union[int, str] = None,
    detector_snapshot_index: Union[int, str] = None,
    device: str = None,
    destfolder: str = None,
    batch_size: int = None,
    detector_batch_size: int = None,
    dynamic: tuple = (False, 0.5, 10),
    ctd_conditions: dict = None,
    ctd_tracking: bool = False,
    top_down_dynamic: dict = None,
    modelprefix: str = "",
    use_shelve: bool = False,
    robust_nframes: bool = False,
    transform = None,
    auto_track: bool = True,
    n_tracks: int = None,
    animal_names: List[str] = None,
    calibrate: bool = False,
    identity_only: bool = None,
    overwrite: bool = False,
    cropping: List[int] = None,
    save_as_df: bool = False,
    bbox_threshold: float = 0.1,
    pose_threshold: float = 0.4,  # Add pose threshold parameter
    model_snapshot_path: str = None,
    detector_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
) -> str:
    """
    Wrapper function that uses the dedicated superanimal_humanbody implementation.
    
    This function mimics the interface of the standard analyze_videos function
    but uses our dedicated implementation for superanimal_humanbody.
    """
    
    # Load model configuration using the standard function (which handles detector config)
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config
    
    # Use the standard function to get the complete config with detector
    model_config = load_super_animal_config(
        super_animal="superanimal_humanbody",
        model_name="rtmpose_x",
        detector_name=detector_name,
        max_individuals=10,  # Default value
        device=device
    )
    
    # Use provided model snapshot path or get it from dlclibrary
    if model_snapshot_path is None:
        from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
        
        # Get the model snapshot path using dlclibrary
        model_snapshot_path = get_super_animal_snapshot_path(
            dataset="superanimal_humanbody",
            model_name="rtmpose_x",
            download=True
        )
    
    # Convert videos to list
    if isinstance(videos, str):
        videos = [videos]
    
    # Set destination folder
    if destfolder is None:
        destfolder = Path(videos[0]).parent
    else:
        destfolder = Path(destfolder)
    
    if not destfolder.exists():
        destfolder.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for video_path in videos:
        print(f"Processing video {video_path}")
        video_name = Path(video_path).stem
        # Use detector_name in scorer and output file names
        dlc_scorer = get_super_animal_scorer(
            "superanimal_humanbody", model_snapshot_path, detector_name
        )
        output_prefix = f"{video_name}_{dlc_scorer}"
        output_json = destfolder / f"{output_prefix}_before_adapt.json"

        if output_json.exists():
            print(f"Predictions already exist for {video_path}, skipping inference.")
            # Load predictions from existing JSON file
            with open(output_json, "r") as f:
                predictions = json.load(f)
            results[video_path] = predictions
        else:
            # Run our dedicated inference
            predictions = video_inference_superanimal_humanbody(
                video=video_path,
                model_config=model_config,
                model_snapshot_path=model_snapshot_path,
                max_individuals=len(model_config["metadata"]["individuals"]),
                bbox_threshold=bbox_threshold,
                device=device,
                cropping=cropping,
                dest_folder=str(destfolder),
            )
            with open(output_json, "w") as f:
                json.dump(predictions, f, cls=NumpyEncoder, indent=2)
            print(f"Results saved to {output_json}")
            results[video_path] = predictions

        # Always create labeled video, regardless of whether predictions already existed
        # Create labeled video just like other superanimal_* models
        # Note: This always runs regardless of whether predictions were loaded or newly created
        try:
            from deeplabcut.pose_estimation_pytorch.apis.videos import create_df_from_prediction
            from deeplabcut.utils.make_labeled_video import create_video
            
            # Convert our predictions to the format expected by create_df_from_prediction
            def convert_predictions_format(predictions, model_config):
                """Convert our prediction format to the format expected by create_df_from_prediction."""
                bodyparts = model_config['metadata']['bodyparts']
                individuals = model_config['metadata'].get('individuals', ['individual_0'])
                
                converted_predictions = []
                for frame_pred in predictions:
                    # Create the expected numpy array: (num_individuals, num_bodyparts, 3)
                    num_individuals = len(individuals)
                    num_bodyparts = len(bodyparts)
                    
                    # Initialize with NaN values
                    bodyparts_array = np.full((num_individuals, num_bodyparts, 3), np.nan)
                    
                    # Handle different prediction formats
                    if 'bodyparts' in frame_pred:
                        if isinstance(frame_pred['bodyparts'], list):
                            # Handle list format (from JSON loading)
                            for i, individual_preds in enumerate(frame_pred['bodyparts']):
                                if i < num_individuals and isinstance(individual_preds, list):
                                    for j, pred in enumerate(individual_preds):
                                        if j < num_bodyparts and len(pred) >= 3:
                                            bodyparts_array[i, j] = [pred[0], pred[1], pred[2]]
                        elif isinstance(frame_pred['bodyparts'], np.ndarray):
                            # Handle numpy array format (from fresh predictions after postprocessing)
                            poses = frame_pred['bodyparts']
                            if poses.shape[1] == num_bodyparts:
                                # poses shape: (num_individuals, num_bodyparts, 3)
                                num_detected = min(poses.shape[0], num_individuals)
                                bodyparts_array[:num_detected] = poses[:num_detected]
                    elif 'bodypart' in frame_pred and 'poses' in frame_pred['bodypart']:
                        # Handle pose runner format (fresh predictions before postprocessing)
                        poses = frame_pred['bodypart']['poses']
                        if isinstance(poses, np.ndarray) and poses.shape[1] == num_bodyparts:
                            # poses shape: (num_individuals, num_bodyparts, 3)
                            num_detected = min(poses.shape[0], num_individuals)
                            bodyparts_array[:num_detected] = poses[:num_detected]
                    
                    # Create the converted prediction
                    converted_pred = {
                        'bodyparts': bodyparts_array
                    }
                    
                    # Add bbox info if available
                    if 'bboxes' in frame_pred:
                        converted_pred['bboxes'] = frame_pred['bboxes']
                    if 'bbox_scores' in frame_pred:
                        converted_pred['bbox_scores'] = frame_pred['bbox_scores']
                    
                    converted_predictions.append(converted_pred)
                
                return converted_predictions
            
            # Convert predictions to the expected format
            converted_predictions = convert_predictions_format(predictions, model_config)
            
            # Get the proper scorer name
            dlc_scorer = get_super_animal_scorer(
                "superanimal_humanbody", model_snapshot_path, detector_name
            )
            
            output_path = destfolder
            output_h5 = output_path / f"{output_prefix}.h5"
            
            # Convert predictions to DataFrame format
            df = create_df_from_prediction(
                predictions=converted_predictions,
                dlc_scorer=dlc_scorer,
                multi_animal=True,
                model_cfg=model_config,
                output_path=output_path,
                output_prefix=output_prefix,
            )
            
            # Save HDF5 file
            df.to_hdf(output_h5, key='df_with_missing', mode='w')
            print(f"Created HDF5 file: {output_h5}")
            
            # Create labeled video using the same approach as other superanimal models
            output_video = output_path / f"{output_prefix}_labeled.mp4"
            
            # Get colormap for humanbody
            superanimal_colormaps = get_superanimal_colormaps()
            colormap = superanimal_colormaps.get("superanimal_humanbody", "rainbow")
            
            # Load skeleton from the superanimal_humanbody.yaml config
            skeleton_edges = None
            try:
                import yaml
                import os
                # Get the correct path to the config file using DeepLabCut's path resolution
                from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path
                dlc_root_path = get_deeplabcut_path()
                config_path = os.path.join(dlc_root_path, "modelzoo", "project_configs", "superanimal_humanbody.yaml")
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                skeleton_indices = config.get('skeleton', None)
                if skeleton_indices:
                    # Convert skeleton indices to bodypart names
                    bodyparts = model_config['metadata']['bodyparts']
                    skeleton_edges = []
                    for idx1, idx2 in skeleton_indices:
                        # Fix 1-based indexing (subtract 1 to convert to 0-based)
                        idx1_0based = idx1 - 1 if idx1 > 0 else idx1
                        idx2_0based = idx2 - 1 if idx2 > 0 else idx2
                        if idx1_0based < len(bodyparts) and idx2_0based < len(bodyparts) and idx1_0based >= 0 and idx2_0based >= 0:
                            skeleton_edges.append((bodyparts[idx1_0based], bodyparts[idx2_0based]))
                        else:
                            print(f"Warning: Skeleton indices {idx1}->{idx1_0based}, {idx2}->{idx2_0based} out of range for {len(bodyparts)} bodyparts")
                    print(f"Loaded skeleton with {len(skeleton_edges)} connections")
                else:
                    print("No skeleton found in config, skeleton plotting will be disabled")
            except Exception as e:
                print(f"Could not load skeleton from config: {e}")
                skeleton_edges = None
            
            # Get bbox info for video creation
            bbox_keys_in_predictions = {"bboxes", "bbox_scores"}
            bboxes_list = [
                {key: value for key, value in p.items() if key in bbox_keys_in_predictions}
                for p in predictions
            ]
            
            # Get cropping info
            bbox = cropping if cropping is not None else (0, 1920, 0, 1080)  # Default bbox
            
            print(f"Creating labeled video for {video_path}...")
            create_video(
                video_path,
                output_h5,
                pcutoff=pose_threshold,
                fps=30,  # Default fps
                bbox=bbox,
                cmap=colormap,
                output_path=str(output_video),
                plot_bboxes=True,
                bboxes_list=bboxes_list,
                bboxes_pcutoff=bbox_threshold,
                skeleton_edges=skeleton_edges,  # Add skeleton support
            )
            print(f"Labeled video created: {output_video}")
            
        except Exception as e:
            print(f"[Warning] Could not create labeled video for {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return str(destfolder)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj) 