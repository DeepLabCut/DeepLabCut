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

import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import deeplabcut.pose_estimation_pytorch.data as data
import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
from deeplabcut.core.engine import Engine
from deeplabcut.modelzoo.utils import get_superanimal_colormaps
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_predictions_dataframe,
    get_inference_runners,
    get_model_snapshots,
    get_scorer_name,
    get_scorer_uid,
    parse_snapshot_index_for_analysis,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import update_config
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.utils import resolve_device
from deeplabcut.utils import auxfun_videos, auxiliaryfunctions


def superanimal_analyze_images(
    superanimal_name: str,
    model_name: str,
    detector_name: str,
    images: str | Path | list[str] | list[Path],
    max_individuals: int,
    out_folder: str | Path,
    bbox_threshold: float = 0.6,
    progress_bar: bool = True,
    device: str | None = None,
) -> dict[str, dict]:
    """
    This funciton inferences a superanimal model on a set of images and saves the
    results as labeled images.

    Args:
        superanimal_name: The name of the superanimal to analyze. Supported list:
            - "superanimal_bird"
            - "superanimal_topviewmouse"
            - "superanimal_quadruped"
        model_name: The name of the pose model architecture to use for inference.
        detector_name: The name of the detector architecture to use for inference.
        images: The images to analyze. Can either be a directory containing images, or
            a list of paths of images.
        max_individuals: The maximum number of individuals to detect in each image.
        out_folder: The directory where the labeled images will be saved.
        bbox_threshold: The minimum confidence score to keep bounding box detections.
            Must be in (0, 1).
        progress_bar: Whether to display a progress bar when running inference.
        device: The device to use to run image analysis.

    Returns:
        The predictions for each image

    Examples:
        >>> import deeplabcut
        >>> from deeplabcut.pose_estimation_pytorch.apis.analyze_images import (
        >>>     superanimal_analyze_images
        >>> )
        >>> superanimal_name = "superanimal_quadruped"
        >>> model_name = "hrnetw32"
        >>> device = "cuda:0"
        >>> max_individuals = 3
        >>> test_images_folder = "test_rodent_images"
        >>> out_images_folder = "vis_test_rodent_images"
        >>> ret = superanimal_analyze_images(
        >>>     superanimal_name,
        >>>     model_name,
        >>>     test_images_folder,
        >>>     max_individuals,
        >>>     out_images_folder
        >>> )
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    snapshot_path = modelzoo.get_super_animal_snapshot_path(
        dataset=superanimal_name, model_name=model_name,
    )
    detector_path = modelzoo.get_super_animal_snapshot_path(
        dataset=superanimal_name, model_name=detector_name,
    )

    config = modelzoo.load_super_animal_config(
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
    )
    config = update_config(config, max_individuals, device)
    config["metadata"]["individuals"] = [f"animal{i}" for i in range(max_individuals)]
    if "detector" in config:
        config["detector"]["model"]["box_score_thresh"] = bbox_threshold

    predictions = analyze_image_folder(
        model_cfg=config,
        images=images,
        snapshot_path=snapshot_path,
        detector_path=detector_path,
        max_individuals=max_individuals,
        device=device,
        progress_bar=progress_bar,
    )

    superanimal_colormaps = get_superanimal_colormaps()
    colormap = superanimal_colormaps[superanimal_name]
    create_labeled_images_from_predictions(predictions, out_folder, colormap)
    return predictions


def analyze_images(
    config: str | Path,
    images: str | Path | list[str] | list[Path],
    frame_type: str | None = None,
    output_dir: str | Path | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    modelprefix: str = "",
    device: str | None = None,
    max_individuals: int | None = None,
    save_as_csv: bool = False,
    progress_bar: bool = True,
) -> dict[str, dict]:
    """Runs analysis on images using a pose model.

    Args:
        config: The project configuration file.
        images: The image(s) to run inference on. Can be the path to an image, the path
            to a directory containing images, or a list of image paths or directories
            containing images.
        frame_type: Filters the images to analyze to only the ones with the given suffix
            (e.g. setting `frame_type`=".png" will only analyze ".png" images). The
            default behavior analyzes all ".jpg", ".jpeg" and ".png" images.
        output_dir: The directory where the predictions will be stored.
        shuffle: The shuffle for which to run image analysis.
        trainingsetindex: The trainingsetindex for which to run image analysis.
        snapshot_index: The index of the snapshot to use. Loaded from the project
            configuration file if None.
        detector_snapshot_index: For top-down models only. The index of the detector
            snapshot to use. Loaded from the project configuration file if None.
        modelprefix: The model prefix used for the shuffle.
        device: The device to use to run image analysis.
        max_individuals: The maximum number of individuals to detect in each image. Set
            to the number of individuals in the project if None.
        save_as_csv: Whether to also save the predictions as a CSV file.
        progress_bar: Whether to display a progress bar when running inference.

    Returns:
        A dictionary mapping each image filename to the different types of predictions
        for it (e.g. "bodyparts", "unique_bodyparts", "bboxes", "bbox_scores")
    """
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][trainingsetindex]
    model_folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_model_folder(
        train_frac,
        shuffle,
        cfg,
        engine=Engine.PYTORCH,
        modelprefix=modelprefix,
    )
    train_folder = model_folder / "train"

    model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
    model_cfg = config_utils.read_config_as_dict(model_cfg_path)
    pose_task = Task(model_cfg["method"])

    # get the snapshots to analyze images with
    snapshot_index, detector_snapshot_index = parse_snapshot_index_for_analysis(
        cfg, model_cfg, snapshot_index, detector_snapshot_index
    )
    snapshot = get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
    detector_snapshot = None
    if detector_snapshot_index is not None:
        detector_snapshot = get_model_snapshots(
            detector_snapshot_index, train_folder, Task.DETECT
        )[0]

    predictions = analyze_image_folder(
        model_cfg=model_cfg,
        images=images,
        snapshot_path=snapshot.path,
        detector_path=None if detector_snapshot is None else detector_snapshot.path,
        frame_type=frame_type,
        device=device,
        max_individuals=max_individuals,
        progress_bar=progress_bar,
    )

    if len(predictions) == 0:
        print(f"Found no images in {images}")
        return {}

    if output_dir is None:
        images = list(predictions.keys())
        output_dir = Path(images[0]).parent.resolve()
        print(f"Setting output directory to {output_dir}")
    output_dir = Path(output_dir)

    scorer = get_scorer_name(
        cfg,
        shuffle=shuffle,
        train_fraction=train_frac,
        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )
    individuals = model_cfg["metadata"]["individuals"]
    if max_individuals is not None:
        individuals = [f"individual{i}" for i in range(max_individuals)]

    df_predictions = build_predictions_dataframe(
        scorer=scorer,
        predictions=predictions,
        parameters=data.PoseDatasetParameters(
            bodyparts=model_cfg["metadata"]["bodyparts"],
            unique_bpts=model_cfg["metadata"]["unique_bodyparts"],
            individuals=individuals,
        ),
        image_name_to_index=None,
    )

    output_filepath = output_dir / f"image_predictions_{scorer}.h5"
    print(f"Saving predictions to {output_filepath}")

    df_predictions.to_hdf(output_filepath, key="predictions")
    if save_as_csv:
        print(f"Saving CSV as {output_filepath}")
        df_predictions.to_csv(output_filepath.with_suffix(".csv"))

    return predictions


def analyze_image_folder(
    model_cfg: str | Path | dict,
    images: str | Path | list[str] | list[Path],
    snapshot_path: str | Path,
    detector_path: str | Path | None = None,
    frame_type: str | None = None,
    device: str | None = None,
    max_individuals: int | None = None,
    progress_bar: bool = True,
) -> dict[str, dict[str, np.ndarray | np.ndarray]]:
    """Runs pose inference on a folder of images

    Args:
        model_cfg: The model config (or its path) used to analyze the images.
        images: The images to analyze. Can either be a directory containing images, or
            a list of paths of images.
        snapshot_path: The path of the snapshot to use to analyze the images.
        detector_path: The path of the detector snapshot to use to analyze the images,
            if a top-down model was used.
        frame_type: Filters the images to analyze to only the ones with the given suffix
            (e.g. setting `frame_type`=".png" will only analyze ".png" images). The
            default behavior analyzes all ".jpg", ".jpeg" and ".png" images.
        device: The device to use to run image analysis.
        max_individuals: The maximum number of individuals to detect in each image. Set
            to the number of individuals in the project if None.
        progress_bar: Whether to display a progress bar when running inference.

    Returns:
        A dictionary mapping each image filename to the different types of predictions
        for it (e.g. "bodyparts", "unique_bodyparts", "bboxes", "bbox_scores")

    Raises:
        ValueError: if the pose model is a top-down model but no detector path is given
    """
    if not isinstance(model_cfg, dict):
        model_cfg = config_utils.read_config_as_dict(model_cfg)

    pose_task = Task(model_cfg["method"])
    if pose_task == Task.TOP_DOWN and detector_path is None:
        raise ValueError(
            "A detector path must be specified for image analysis using top-down models"
            f" Please specify the `detector_path` parameter."
        )

    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    individuals = model_cfg["metadata"]["individuals"]
    if max_individuals is None:
        max_individuals = len(individuals)

    if device is None:
        device = resolve_device(model_cfg)

    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(bodyparts),
        num_unique_bodyparts=len(unique_bodyparts),
        device=device,
        with_identity=False,
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )

    image_suffixes = ".png", ".jpg", ".jpeg"
    if frame_type is not None:
        image_suffixes = (frame_type, )

    image_paths = parse_images_and_image_folders(images, image_suffixes)
    pose_inputs = image_paths
    if detector_runner is not None:
        logging.info(f"Running object detection with {detector_path}")

        detector_image_paths = image_paths
        if progress_bar:
            detector_image_paths = tqdm(detector_image_paths)
        bbox_predictions = detector_runner.inference(images=detector_image_paths)
        pose_inputs = list(zip(image_paths, bbox_predictions))

    logging.info(f"Running pose estimation with {detector_path}")

    if progress_bar:
        pose_inputs = tqdm(pose_inputs)

    predictions = pose_runner.inference(pose_inputs)

    return {
        image_path: image_predictions
        for image_path, image_predictions in zip(image_paths, predictions)
    }


def create_labeled_images_from_predictions(predictions, out_folder, cmap):
    for image_path, prediction in predictions.items():
        frame = auxfun_videos.imread(str(image_path), mode="skimage")
        fig, ax = plt.subplots()
        ax.imshow(frame)
        for idx, pose in enumerate(prediction["bodyparts"]):
            x, y, confidence = pose[:, 0], pose[:, 1], pose[:, 2]
            if np.sum(pose) < 0:
                continue
            mask = confidence > 0.0
            x = x[mask]
            y = y[mask]
            ax.scatter(x, y, c=np.arange(len(x)), cmap=cmap)
        bboxes = prediction["bboxes"]
        for bbox in bboxes:
            # Draw bounding boxes around detected objects
            xmin, ymin, w, h = bbox
            rect = plt.Rectangle(
                (xmin, ymin), w, h, fill=False, edgecolor="green", linewidth=2
            )

            ax.add_patch(rect)
        image_name = image_path.split(os.sep)[-1]
        fig.savefig(os.path.join(out_folder, f"vis_{image_name}"))


def plot_images_coco(
    model_cfg: str | Path | dict,
    image_folder: str | Path,
    snapshot_path: str | Path,
    out_path: str = "test_images",
    data_json_path: str = "",
    detector_path: str | Path | None = None,
    device: str | None = None,
    max_individuals: int | None = None,
) -> list[dict]:
    """
    Runs pose inference on a folder of images from a COCO dataset, and plots all
    predicted keypoints and bounding boxes

    Args:
        model_cfg: The model config (or its path) used to analyze the images.
        image_folder: The path to the folder containing the images to analyze.
        snapshot_path: The path of the snapshot to use to analyze the images.
        out_path: The path of the folder where images should be output.
        data_json_path: The path to the JSON file containing ground truth data.
        detector_path: The path of the detector snapshot to use to analyze the images,
            if a top-down model was used.
        device: The device on which to run image inference
        max_individuals: The maximum number of individuals to detect in an image.

    Returns:
        A list of dictionaries containing predictions made on each image.

    Raises:
        ValueError: if a top-down model configuration is given but detector_path is None
    """
    with open(data_json_path, "r") as f:
        obj = json.load(f)

    coco_images = obj["images"]
    coco_annotations = obj["annotations"]

    image_name_to_id = {}
    for image in coco_images:
        # only works with relative path as a test image can be in a different folder
        image_name = image["file_name"].split(os.sep)[-1]
        image_name_to_id[image_name] = image["id"]

    image_id_to_annotations = defaultdict(list)
    image_ids = list(image_name_to_id.values())
    for annotation in coco_annotations:
        image_id = annotation["image_id"]
        if annotation["image_id"] in image_ids:
            image_id_to_annotations[image_id].append(annotation)

    # need to support more image types
    images_in_folder = glob.glob(str(Path(image_folder) / "*.png"))
    corresponded_images = []
    for image in images_in_folder:
        image_path = image
        image_name = image.split(os.sep)[-1]
        if image_name in image_name_to_id:
            corresponded_images.append(image_path)

    images = corresponded_images

    predictions = analyze_image_folder(
        model_cfg=model_cfg,
        images=images,
        snapshot_path=snapshot_path,
        detector_path=detector_path,
        device=device,
        max_individuals=max_individuals,
        progress_bar=True,
    )

    os.makedirs(out_path, exist_ok=True)

    coco_format_predictions = []
    for image_path, prediction in predictions.items():
        image_name = image_path.split(os.sep)[-1]
        coco_prediction = dict(
            image_id=image_name_to_id[image_name],
            gt_annotations=image_id_to_annotations[image_name_to_id[image_name]],
            file_name=image_path,
            bodyparts=prediction["bodyparts"],
        )
        if "unique_bodyparts" in prediction:
            coco_prediction["unique_bodyparts"] = prediction["unique_bodyparts"]
        if "bboxes" in prediction:
            coco_prediction["bboxes"] = prediction["bboxes"]
        if "bbox_scores" in prediction:
            coco_prediction["bbox_scores"] = prediction["bbox_scores"]

        coco_format_predictions.append(coco_prediction)

        frame = auxfun_videos.imread(str(image_path), mode="skimage")
        fig, ax = plt.subplots()
        ax.imshow(frame)

        # TODO: color of keypoints are all red. Need to change to a different colormap
        for pose in prediction["bodyparts"]:
            x, y, confidence = pose[:, 0], pose[:, 1], pose[:, 2]
            mask = confidence > 0.0
            x = x[mask]
            y = y[mask]
            ax.scatter(x, y, color="red")

        bboxes = prediction["bboxes"]
        for bbox in bboxes:
            # Draw bounding boxes around detected objects
            xmin, ymin, w, h = bbox
            rect = plt.Rectangle(
                (xmin, ymin), w, h, fill=False, edgecolor="blue", linewidth=2
            )

        ax.add_patch(rect)
        image_name = image_path.split("/")[-1]
        fig.savefig(os.path.join(out_path, image_name))

    return coco_format_predictions


def parse_images_and_image_folders(
    images: str | Path | list[str] | list[Path],
    image_suffixes: tuple[str] = (".png", ".jpg", ".jpeg"),
) -> list[str]:
    """Parses image paths or directory paths into a single list of image paths.

    Args:
        images: Paths of images or folders containing images.
        image_suffixes: Suffixes used for images.

    Returns:
        The images contained in the folders or directly the paths given as input
    """
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            return [str(img) for img in path.iterdir() if img.suffix in image_suffixes]

        return [str(path)]

    image_to_analyze = []
    for file in images:
        image_to_analyze += parse_images_and_image_folders(file)

    return image_to_analyze
