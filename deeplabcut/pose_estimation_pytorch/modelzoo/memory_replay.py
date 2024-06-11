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
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.modelzoo.generalized_data_converter.datasets import (
    COCOPoseDataset,
    MaDLCPoseDataset,
    MultiSourceDataset,
    SingleDLCPoseDataset,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_config_model_paths,
    update_config,
)
from deeplabcut.utils.pseudo_label import calculate_iou, optimal_match, xywh2xyxy


# this is reading from a coco project
def prepare_memory_replay_dataset(
    source_dataset_folder,
    superanimal_name,
    model_name,
    max_individuals=1,
    train_file="train.json",
    test_file="test.json",
    pose_threshold=0.0,
    device=None,
    pose_model_path="",
    detector_path="",
    customized_pose_checkpoint=None,
):
    """
    Need to first run inference on the source project train file
    """

    (
        model_config,
        project_config,
        pose_model_path,
        detector_path,
    ) = get_config_model_paths(superanimal_name, model_name)

    if customized_pose_checkpoint is not None:
        print(
            "memory replay fine-tuning pose checkpoint is replaced by",
            customized_pose_checkpoint,
        )

    config = {**project_config, **model_config}
    config = update_config(config, max_individuals, device)
    individuals = [f"animal{i}" for i in range(max_individuals)]
    config["individuals"] = individuals
    num_bodyparts = len(config["bodyparts"])
    train_file_path = os.path.join(source_dataset_folder, "annotations", train_file)

    pose_runner, detector_runner = get_inference_runners(
        config,
        snapshot_path=pose_model_path,
        max_individuals=max_individuals,
        num_bodyparts=len(model_config["metadata"]["bodyparts"]),
        num_unique_bodyparts=0,
        detector_path=detector_path,
    )

    with open(train_file_path, "r") as f:
        train_obj = json.load(f)

    images = train_obj["images"]
    annotations = train_obj["annotations"]
    categories = train_obj["categories"]
    imagename2id = {}
    imageid2name = {}
    imagename2gt = defaultdict(list)

    for image in images:
        # this only works with relative path as the testing image can be at a different folder
        imagename = image["file_name"].split(os.sep)[-1]
        imagename2id[imagename] = image["id"]
        imageid2name[image["id"]] = imagename

    imagename2bbox = defaultdict(list)
    for anno in annotations:
        imagename = imageid2name[anno["image_id"]]
        imagename2gt[imagename].append(anno)
        imagename2bbox[imagename].append(anno["bbox"])

    imageid2annotations = defaultdict(list)

    imageids = list(imagename2id.values())
    for annotation in annotations:
        image_id = annotation["image_id"]
        if annotation["image_id"] in imageids:
            imageid2annotations[image_id].append(annotation)

    # need to support more image types
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"]

    images_in_folder = []
    for ext in image_extensions:
        images_in_folder.extend(
            glob.glob(os.path.join(source_dataset_folder, "images", ext))
        )

    corresponded_images = []
    for image in images_in_folder:
        image_path = image
        imagename = image.split(os.sep)[-1]
        if imagename in imagename2id:
            corresponded_images.append(image_path)

    images = corresponded_images

    bbox_predictions = detector_runner.inference(images=images)

    bbox_gts = [
        {"bboxes": np.array(imagename2bbox[image.split(os.sep)[-1]])}
        for image in images
    ]

    pose_inputs = list(zip(images, bbox_gts))

    # pose inference should return meta data for pseudo labeling
    predictions = pose_runner.inference(pose_inputs)

    assert len(images) == len(predictions)

    imagename2prediction = {}

    for image_path, prediction in zip(images, predictions):
        imagename = image_path.split(os.sep)[-1]
        imagename2prediction[imagename] = prediction

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

    for imagename, gts in imagename2gt.items():
        bbox_gts = [np.array(gt["bbox"]) for gt in gts]
        bbox_gts = [xywh2xyxy(e) for e in bbox_gts]
        prediction = imagename2prediction[imagename]
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
                    if (
                        matched_gt[kpt_idx][2] < pose_threshold
                        and matched_gt[kpt_idx][2] > 0
                    ):
                        matched_gt[kpt_idx][2] = -1
                    elif matched_gt[kpt_idx][2] > 0:
                        matched_gt[kpt_idx][2] = 2

                gts[idx]["keypoints"] = list(matched_gt.flatten())

    # memory replay path
    memory_replay_train_file_path = os.path.join(
        source_dataset_folder, "annotations", "memory_replay_train.json"
    )

    with open(memory_replay_train_file_path, "w") as f:
        json.dump(train_obj, f, indent=4)


def prepare_memory_replay(
    dlc_proj_root: str | Path,
    shuffle: int,
    superanimal_name: str,
    model_name: str,
    device: str,
    max_individuals=3,
    trainingsetindex: int = 0,
    train_file="train.json",
    pose_threshold=0.1,
    customized_pose_checkpoint=None,
):
    """TODO: Documentation"""

    # in order to fill the num_bodyparts stuff

    config_path = Path(dlc_proj_root, "config.yaml")
    cfg = af.read_config(config_path)

    if "individuals" in cfg:
        temp_dataset = MaDLCPoseDataset(
            str(dlc_proj_root), "temp_dataset", shuffle=shuffle
        )
    else:
        temp_dataset = SingleDLCPoseDataset(
            str(dlc_proj_root), "temp_dataset", shuffle=shuffle
        )

    dlc_proj_root = Path(dlc_proj_root)
    config_path = dlc_proj_root / "config.yaml"

    cfg = af.read_config(config_path)

    trainIndex = 0

    model_folder = dlc_proj_root / af.get_model_folder(
        cfg["TrainingFraction"][trainIndex], shuffle, cfg, engine=Engine.PYTORCH
    )

    memory_replay_folder = model_folder / "memory_replay"

    temp_dataset.materialize(
        memory_replay_folder, framework="coco", append_image_id=False
    )

    original_model_config = af.read_config(
        str(model_folder / "train" / "pytorch_config.yaml")
    )

    weight_init_cfg = original_model_config["train_settings"].get("weight_init")
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

    conversion_table_path = dlc_proj_root / "memory_replay" / "conversion_table.csv"

    # here we project the original DLC projects to superanimal space and save them into a coco project format
    dataset.project_with_conversion_table(str(conversion_table_path))
    dataset.materialize(memory_replay_folder, deepcopy=False, framework="coco")

    # then in this function, we do pseudo label to match prediction and gts to create memory-replay dataset that will be named memory_replay_train.json
    memory_replay_train_file = os.path.join(
        memory_replay_folder, "annotations", "memory_replay_train.json"
    )

    prepare_memory_replay_dataset(
        memory_replay_folder,
        superanimal_name,
        model_name,
        max_individuals=max_individuals,
        device=device,
        train_file=train_file,
        pose_threshold=pose_threshold,
        customized_pose_checkpoint=customized_pose_checkpoint,
    )
