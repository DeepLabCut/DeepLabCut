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

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.modelzoo.generalized_data_converter.datasets import (
    COCOPoseDataset,
    MaDLCPoseDataset,
    SingleDLCPoseDataset,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
from deeplabcut.pose_estimation_pytorch.modelzoo import (
    get_super_animal_project_config_path,
)
from deeplabcut.utils.pseudo_label import calculate_iou


def get_pose_predictions(
    loader: DLCLoader,
    images: list[str],
    bboxes: dict[str, list],
    superanimal_name: str,
    model_snapshot_path: str | Path,
    detector_snapshot_path: str | Path,
    max_individuals: int,
    device: str | None = None,
) -> dict[str, dict]:
    """Gets predictions made by a SuperAnimal model on a DeepLabCut project

    Args:
        loader: The path to the root of the project.
        images: The images on which to run inference with the SuperAnimal model.
        bboxes: The ground truth bounding boxes for each image in the project.
        superanimal_name: The name of the SuperAnimal dataset being used.
        model_snapshot_path: The path to the SuperAnimal pose snapshot.
        detector_snapshot_path: The path to the SuperAnimal detector snapshot.
        max_individuals: The maximum number of individuals to detect per image.
        device: The CUDA device to use.

    Returns:
        The predictions made by the SuperAnimal model on each image in the images list.
    """
    model_name = detector_snapshot_path.stem + "-" + model_snapshot_path.stem
    predictions_folder = (
        loader.project_path / "memory_replay" / superanimal_name / model_name
    )
    predictions_folder.mkdir(exist_ok=True, parents=True)
    predictions_file = predictions_folder / "pseudo-labels.json"

    # COCO-format annotations file containing predictions made by the SuperAnimal model
    sa_predictions = {}
    if predictions_file.exists():
        with open(predictions_file, "r") as f:
            raw_sa_predictions = json.load(f)

        # parse predictions to convert lists to numpy arrays
        for image, predictions in raw_sa_predictions.items():
            sa_predictions[image] = {
                "bodyparts": np.array(predictions["bodyparts"]),
                "bboxes": np.array(predictions["bboxes"]),
                # "bbox_scores": np.array(predictions["bbox_scores"]),
            }

    # get images that need to be processed
    processed_images = set(sa_predictions.keys())
    images_to_process = [image for image in (set(images) - processed_images)]

    # if all images have been processed by the SuperAnimal model, return the predictions
    if len(images_to_process) == 0:
        return sa_predictions

    pose_runner, detector_runner = get_inference_runners(
        loader.model_cfg,
        snapshot_path=model_snapshot_path,
        max_individuals=max_individuals,
        num_bodyparts=len(loader.model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(loader.model_cfg["metadata"]["unique_bodyparts"]),
        device=device,
        detector_path=detector_snapshot_path,
    )

    # FIXME(niels, yeshaokai) - Use the detector to combine GT-keypoint created bounding
    #  boxes and predicted bounding boxes - keep the larger of the two
    # bbox_predictions = detector_runner.inference(images=images_to_process)
    pose_inputs = [
        (
            str(loader.project_path / Path(image)),
            {"bboxes": np.array(bboxes[image])}
        )
        for image in images_to_process
    ]
    predictions = pose_runner.inference(pose_inputs)

    for image, prediction in zip(images_to_process, predictions):
        sa_predictions[image] = prediction

    # save the updated SuperAnimal predictions
    json_sa_predictions = {
        image: {
            "bodyparts": predictions["bodyparts"].tolist(),
            "bboxes": predictions["bboxes"].tolist(),
            # "bbox_scores": predictions["bbox_scores"].tolist(),
        }
        for image, predictions in sa_predictions.items()
    }
    with open(predictions_file, "w") as f:
        json.dump(json_sa_predictions, f, indent=2)

    return sa_predictions


# this is reading from a coco project
def prepare_memory_replay_dataset(
    loader: DLCLoader,
    source_dataset_folder: str | Path,
    superanimal_name: str,
    model_snapshot_path: str,
    detector_snapshot_path: str,
    max_individuals: int = 1,
    train_file: str = "train.json",
    pose_threshold: float = 0.0,
    device: str | None = None,
):
    """
    Need to first run inference on the source project train file
    """
    project_root = loader.project_path.resolve()
    source_dataset_folder = Path(source_dataset_folder).resolve()

    # Contains the ground truth annotations for the DeepLabCut project
    # .../dlc-models-pytorch/.../...shuffle0/train/memory_replay/annotations/train.json
    with open(source_dataset_folder / "annotations" / train_file, "r") as f:
        project_gt = json.load(f)

    # parse the GT so that image paths are in the format (no matter the OS):
    # "labeled-data/{video_name}/{image_name}"
    for image in project_gt["images"]:
        image["file_name"] = "/".join(Path(image["file_name"]).parts[-3:])

    image_id_to_name = {}
    image_id_to_annotations = defaultdict(list)

    image_name_to_id = {}
    image_name_to_gt = defaultdict(list)
    image_name_to_bbox = defaultdict(list)

    for image in project_gt["images"]:
        image_name_to_id[image["file_name"]] = image["id"]
        image_id_to_name[image["id"]] = image["file_name"]

    for anno in project_gt["annotations"]:
        name = image_id_to_name[anno["image_id"]]
        image_name_to_gt[name].append(anno)
        image_name_to_bbox[name].append(anno["bbox"])

    image_ids = list(image_name_to_id.values())
    for annotation in project_gt["annotations"]:
        image_id = annotation["image_id"]
        if annotation["image_id"] in image_ids:
            image_id_to_annotations[image_id].append(annotation)

    image_name_to_prediction = get_pose_predictions(
        loader=loader,
        images=[image["file_name"] for image in project_gt["images"]],
        bboxes=image_name_to_bbox,
        superanimal_name=superanimal_name,
        model_snapshot_path=model_snapshot_path,
        detector_snapshot_path=detector_snapshot_path,
        max_individuals=max_individuals,
        device=device,
    )

    def xywh2xyxy(bbox):
        temp_bbox = np.copy(bbox)
        temp_bbox[2:] = temp_bbox[:2] + temp_bbox[2:]
        return temp_bbox

    def optimal_match(gts_list, preds_list):
        arranged_preds_list = []
        num_gts = len(gts_list)
        num_preds = len(preds_list)
        cost_matrix = np.zeros((num_gts, num_preds))

        for i in range(num_gts):
            for j in range(num_preds):
                cost_matrix[i, j] = distance.euclidean(
                    gts_list[i][..., :2].flatten(), preds_list[j][..., :2].flatten()
                )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return col_ind

    num_bodyparts = len(project_gt["categories"][0]["keypoints"])
    for image_name, gts in image_name_to_gt.items():
        bbox_gts = [np.array(gt["bbox"]) for gt in gts]
        bbox_gts = [xywh2xyxy(e) for e in bbox_gts]
        prediction = image_name_to_prediction[image_name]
        bbox_preds = [xywh2xyxy(pred) for pred in prediction["bboxes"]]
        optimal_pred_indices = optimal_match(bbox_gts, bbox_preds)

        for idx in range(len(bbox_gts)):
            if idx == len(optimal_pred_indices):
                break

            optimal_index = optimal_pred_indices[idx]
            matched_gt = np.array(gts[idx]["keypoints"])
            matched_pred = prediction["bodyparts"][optimal_index]
            bbox_gt = bbox_gts[idx]
            bbox_pred = bbox_preds[idx]

            # maybe check iou of two bbox
            iou = calculate_iou(bbox_gt, bbox_pred)
            if iou < 0.7:
                matched_gt = np.ones_like(matched_gt) * -1
                gts[idx]["keypoints"] = list(matched_gt.flatten())
            else:
                matched_gt = matched_gt.reshape(num_bodyparts, -1)
                matched_pred = matched_pred.reshape(num_bodyparts, -1)
                mask = matched_gt == -1
                matched_gt[mask] = matched_pred[mask]
                # after the mixing, we don't care about confidence anymore

                for kpt_idx in range(len(matched_gt)):
                    if 0 < matched_gt[kpt_idx][2] < pose_threshold:
                        matched_gt[kpt_idx][2] = -1
                    elif matched_gt[kpt_idx][2] > 0:
                        matched_gt[kpt_idx][2] = 2

                gts[idx]["keypoints"] = list(matched_gt.flatten())

    # memory replay path
    memory_replay_train_file_path = os.path.join(
        source_dataset_folder, "annotations", "memory_replay_train.json"
    )

    # parse the GT to put the image paths back into OS-specific format
    for image in project_gt["images"]:
        image_rel_path = image["file_name"].split("/")
        image["file_name"] = str(project_root.resolve() / Path(*image_rel_path))

    with open(memory_replay_train_file_path, "w") as f:
        json.dump(project_gt, f, indent=4)


def prepare_memory_replay(
    config: str | Path,
    loader: DLCLoader,
    superanimal_name: str,
    model_snapshot_path: str | Path,
    detector_snapshot_path: str | Path,
    device: str,
    max_individuals: int = 3,
    train_file: str = "train.json",
    pose_threshold: float = 0.1,
) -> None:
    """Prepares a shuffle to be trained with memory replay.

    To be trained using memory replay, predictions must be made on all images in the
    dataset using the SuperAnimal model. Predictions for bodyparts that aren't labeled
    in the DeepLabCut project are then used as pseudo-labels during training.

    This method will create a COCO-format dataset in the same folder as the
    ``pytorch_config.yaml`` (the model folder).

    Args:
        config: Path to the DeepLabCut project configuration file.
        loader: The loader used to load the training/test data on which a model will
            be fine-tuned with memory replay.
        superanimal_name: The name of the SuperAnimal model that is being fine-tuned.
        model_snapshot_path: Path to the SuperAnimal pose snapshot to fine-tune.
        detector_snapshot_path: Path to the SuperAnimal detector snapshot to fine-tune.
        device: Device to use to run inference using the SuperAnimal model.
        max_individuals: Maximum number of animals that can be present in a frame.
        train_file: Name of the file containing train annotations (e.g. `train.json`).
        pose_threshold: The minimum score for a prediction to be used as a pseudo-label.
    """
    cfg = af.read_config(config)
    super_animal_cfg = af.read_plainconfig(
        get_super_animal_project_config_path(super_animal=superanimal_name)
    )

    if "individuals" in cfg:
        temp_dataset = MaDLCPoseDataset(
            str(loader.project_path), "temp_dataset", shuffle=loader.shuffle
        )
    else:
        temp_dataset = SingleDLCPoseDataset(
            str(loader.project_path), "temp_dataset", shuffle=loader.shuffle
        )

    memory_replay_folder = loader.model_folder / "memory_replay"
    temp_dataset.materialize(
        memory_replay_folder,
        framework="coco",
        append_image_id=False,
        no_image_copy=True,  # use the images in the labeled-data folder
    )

    weight_init_cfg = loader.model_cfg["train_settings"].get("weight_init")
    if weight_init_cfg is None:
        raise ValueError(
            "You can only train models with memory replay when you are fine-tuning a "
            "SuperAnimal model. Please look at the documentation to see how to create "
            "a training dataset to fine-tune one of the SuperAnimal models."
        )

    weight_init = WeightInitialization.from_dict(weight_init_cfg)
    if not weight_init.with_decoder:
        raise ValueError(
            "You can only train models with memory replay when you are fine-tuning a "
            "SuperAnimal model. Please look at the documentation to see how to create "
            "a training dataset to fine-tune one of the SuperAnimal models. Ensure "
            "that a conversion table is specified for your project and that you select"
            "``with_decoder=True`` for your ``WeightInitialization``."
        )

    dataset = COCOPoseDataset(memory_replay_folder, "memory_replay_dataset")

    # here we project the original DLC projects to superanimal space and save them into
    # a coco project format
    bodyparts = af.get_bodyparts(cfg)
    sa_bodyparts = af.get_bodyparts(super_animal_cfg)
    conversion_table = {}
    for idx, bpt in enumerate(bodyparts):
        conversion_table[bpt] = sa_bodyparts[weight_init.conversion_array[idx]]

    dataset.project_with_conversion_table(
        table_path=None,
        table_dict=dict(
            master_keypoints=sa_bodyparts,
            conversion_table=conversion_table,
        ),
    )

    dataset.materialize(
        memory_replay_folder, framework="coco", deepcopy=False, no_image_copy=True,
    )

    # then in this function, we do pseudo label to match prediction and gts to create
    # memory-replay dataset that will be named memory_replay_train.json
    prepare_memory_replay_dataset(
        loader,
        memory_replay_folder,
        superanimal_name,
        model_snapshot_path,
        detector_snapshot_path,
        max_individuals=max_individuals,
        device=device,
        train_file=train_file,
        pose_threshold=pose_threshold,
    )
