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

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist

import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.modelzoo.generalized_data_converter.datasets import (
    MaDLCDataFrame,
    SingleDLCDataFrame,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    select_device,
    update_config,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)


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


def calculate_iou(box1, box2):
    # Unpack the coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # Calculate the width and height of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Calculate the area of the intersection rectangle
    inter_area = inter_width * inter_height

    # Calculate the area of each bounding box
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the area of the union of the two bounding boxes
    union_area = area_1 + area_2 - inter_area

    # Calculate the IoU
    iou = inter_area / union_area

    return iou


def video_to_frames(input_video, output_folder, cropping: list[int] | None = None):
    # Create the output folder if it doesn't exist
    video = cv2.VideoCapture(str(input_video))
    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # Initialize a frame counter
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        # Crop the frame if desired
        if cropping is not None:
            x1, x2, y1, y2 = cropping
            frame = frame[y1:y2, x1:x2]

        # Save the frame as an image file.
        frame_str = str(frame_count).zfill(5)
        frame_file = os.path.join(output_folder, "images", f"frame_{frame_str}.png")
        cv2.imwrite(frame_file, frame)
        # Increment the frame counter
        frame_count += 1
    # Release the video object and close the window (if open)
    video.release()
    # cv2.destroyAllWindows()


def plot_cost_matrix(
    matrix, gt_keypoint_names, pred_keypoint_names, conversion_plot_out_path
):

    matrix /= np.max(matrix)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.set_xlim(0, int(matrix.shape[1]))
    ax.set_ylim(0, int(matrix.shape[0]))
    ax.set_yticklabels(pred_keypoint_names, minor=False)
    ax.set_xticklabels(gt_keypoint_names, minor=False)
    ax.set_title("cost matrix")
    plt.xticks(rotation=90)
    fig = plt.gcf()
    fig.tight_layout()

    plt.savefig(conversion_plot_out_path, dpi=300)


def keypoint_matching(
    config_path: str | Path,
    superanimal_name: str,
    model_name: str,
    detector_name: str,
    copy_images: bool = False,
    device: str | None = None,
    train_file: str = "train.json",
):
    """Runs the keypoint matching algorithm for a DeepLabCut project

    Matches project keypoints to SuperAnimal keypoints automatically, by running
    SuperAnimal inference on all images in the dataset

    Args:
        config_path: The path of the DeepLabCut project configuration file.
        superanimal_name: SuperAnimal dataset with which to run keypoint matching.
        model_name: Name of the SuperAnimal pose model architecture with which to run
            keypoint matching
        detector_name: Name of the SuperAnimal detector architecture with which to run
            keypoint matching
        copy_images: When False, symlinks are created for the dataset used for keypoint
            matching. Otherwise, images are copied from the `labeled-data` folder to the
            folder used for keypoint matching.
        device: The device on which to run keypoint matching.
        train_file: The name of the file containing the labels to output.
    """
    config_path = Path(config_path)
    cfg = af.read_config(str(config_path))
    dlc_proj_root = config_path.parent

    if "individuals" in cfg:
        temp_dataset = MaDLCDataFrame(str(dlc_proj_root), "temp_dataset")
        max_individuals = len(cfg["individuals"])
    else:
        temp_dataset = SingleDLCDataFrame(str(dlc_proj_root), "temp_dataset")
        max_individuals = 1

    memory_replay_folder = dlc_proj_root / "memory_replay"
    temp_dataset.materialize(
        str(memory_replay_folder), framework="coco", deepcopy=copy_images
    )

    # run inference on the train set
    config = modelzoo.load_super_animal_config(
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
    )
    if device is None:
        device = select_device()

    # get the SuperAnimal detector and pose model snapshot paths
    pose_model_path = modelzoo.get_super_animal_snapshot_path(
        dataset=superanimal_name, model_name=model_name,
    )
    detector_path = modelzoo.get_super_animal_snapshot_path(
        dataset=superanimal_name, model_name=detector_name,
    )

    config = update_config(config, max_individuals, device)
    individuals = [f"animal{i}" for i in range(max_individuals)]
    config["metadata"]["individuals"] = individuals
    train_file_path = os.path.join(memory_replay_folder, "annotations", train_file)

    pose_runner, detector_runner = get_inference_runners(
        config,
        snapshot_path=pose_model_path,
        max_individuals=max_individuals,
        num_bodyparts=len(config["metadata"]["bodyparts"]),
        num_unique_bodyparts=0,
        detector_path=detector_path,
    )

    with open(train_file_path, "r") as f:
        train_obj = json.load(f)

    images = train_obj["images"]
    annotations = train_obj["annotations"]
    categories = train_obj["categories"]
    image_name_to_id = {}
    image_id_to_name = {}

    image_name_to_gt = defaultdict(list)
    image_name_to_bbox = defaultdict(list)
    image_id_to_annotations = defaultdict(list)

    for image in images:
        # this only works with relative path as the testing image can be at a different folder
        name = image["file_name"].split(os.sep)[-1]
        image_name_to_id[name] = image["id"]
        image_id_to_name[image["id"]] = name

    for anno in annotations:
        name = image_id_to_name[anno["image_id"]]
        image_name_to_gt[name].append(anno)
        image_name_to_bbox[name].append(anno["bbox"])

    image_ids = set(image_name_to_id.values())
    for anno in annotations:
        image_id = anno["image_id"]
        if anno["image_id"] in image_ids:
            image_id_to_annotations[image_id].append(anno)

    # need to support more image types
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"]
    images_in_folder = []
    for ext in image_extensions:
        images_in_folder.extend(
            glob.glob(os.path.join(memory_replay_folder, "images", ext))
        )

    corresponded_images = []
    for image in images_in_folder:
        image_path = image
        name = image.split(os.sep)[-1]
        if name in image_name_to_id:
            corresponded_images.append(image_path)

    images = corresponded_images
    bbox_gts = [
        {"bboxes": np.array(image_name_to_bbox[image.split(os.sep)[-1]])}
        for image in images
    ]

    pose_inputs = list(zip(images, bbox_gts))

    # pose inference should return meta data for pseudo labeling
    predictions = pose_runner.inference(pose_inputs)

    with open(str(memory_replay_folder / "pseudo_predictions.json"), "w") as f:
        json.dump(pose_inputs, f, cls=NumpyEncoder)

    assert len(images) == len(predictions)

    image_name_to_pred = {}
    for image_path, prediction in zip(images, predictions):
        name = image_path.split(os.sep)[-1]
        image_name_to_pred[name] = prediction

    pred_keypoint_names = config["metadata"]["bodyparts"]
    num_pred_keypoints = len(pred_keypoint_names)
    gt_keypoint_names = categories[0]["keypoints"]
    num_gt_keypoints = len(gt_keypoint_names)

    match_matrix = np.zeros((num_pred_keypoints, num_gt_keypoints))
    match_dict = defaultdict(lambda: defaultdict(int))

    for name, gts in image_name_to_gt.items():
        bbox_gts = [np.array(gt["bbox"]) for gt in gts]
        bbox_gts = [xywh2xyxy(e) for e in bbox_gts]
        prediction = image_name_to_pred[name]
        bbox_preds = [xywh2xyxy(pred) for pred in prediction["bboxes"]]
        optimal_pred_indices = optimal_match(bbox_gts, bbox_preds)

        for idx in range(len(bbox_gts)):
            if idx == len(optimal_pred_indices):
                break

            optimal_index = optimal_pred_indices[idx]
            matched_gt = np.array(gts[idx]["keypoints"])
            matched_pred = prediction["bodyparts"][optimal_index]
            matched_gt = matched_gt.reshape(num_gt_keypoints, -1)
            matched_pred = matched_pred.reshape(num_pred_keypoints, -1)

            pair_distance = cdist(matched_pred, matched_gt)
            row_ind, column_ind = linear_sum_assignment(pair_distance)
            for row, column in zip(row_ind, column_ind):
                pred_kpt_name = pred_keypoint_names[row]
                anno_kpt_name = gt_keypoint_names[column]
                match_matrix[row][column] += 1
                match_dict[pred_kpt_name][anno_kpt_name] += 1

    row_ind, column_ind = linear_sum_assignment(match_matrix * -1)
    keypoint_mapping_list = []

    conversion_matrix_out_path = os.path.join(
        memory_replay_folder, "confusion_matrix.png"
    )

    plot_cost_matrix(
        match_matrix, gt_keypoint_names, pred_keypoint_names, conversion_matrix_out_path
    )

    for row, column in zip(row_ind, column_ind):
        pred_kpt_name = pred_keypoint_names[row]
        anno_kpt_name = gt_keypoint_names[column]
        count = match_dict[pred_kpt_name][anno_kpt_name]
        keypoint_mapping_list.append((pred_kpt_name, anno_kpt_name, count))

    keypoint_mapping_list = sorted(
        keypoint_mapping_list, key=lambda x: x[2], reverse=True
    )

    names = [e[:2] for e in keypoint_mapping_list]
    conversion_table = {}
    for pred, anno in names:
        conversion_table[pred] = anno

    conversion_table_out_path = os.path.join(
        memory_replay_folder, "conversion_table.csv"
    )
    with open(conversion_table_out_path, "w") as f:
        out = "gt, MasterName\n"
        for name in pred_keypoint_names:
            target = name
            source = conversion_table.get(target, "")
            out += f"{source}, {target}\n"
        f.write(out)


# this is to generate a coco project as an intermediate data
def dlc3predictions_2_annotation_from_video(
    predictions,
    dest_proj_folder,
    bodyparts,
    superanimal_name,
    pose_threshold=0.0,
    bbox_threshold=0.0,
):
    """
    For video adaptation, we also need to create a coco project
    dlc3 predictions:

    list of dictionary
    [{
    bodyparts:[] # (n_individuals, n_kpts, 3)
    bboxes: [] # (n_individuals, 4) -> x,y,w,h
    }]

    coco result is a list of dictionary
    # i might get a minimal version that works with my script

    category_id:
    image_id: []
    image_path: []
    keypoints: []
    score: []
    bbox: []

    """

    category_id = 1  # the default for superanimal. But it might be changed

    images = []
    annotations = []
    categories = []
    annotation_id = 0
    image_folder = os.path.join(dest_proj_folder, "images")

    # video_to_frames function by default outputs png or jpg
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    # Ensure predictions and image_paths have the same length before subsampling
    if len(predictions) != len(image_paths):
        print(f"Warning: predictions length ({len(predictions)}) != image_paths length ({len(image_paths)})")
        # Take the minimum length to avoid index errors
        min_length = min(len(predictions), len(image_paths))
        predictions = predictions[:min_length]
        image_paths = image_paths[:min_length]
        print(f"Truncated both arrays to length {min_length}")

    # skipping every 10 frames should speed up and not impact the performance
    predictions, image_paths = predictions[::10], image_paths[::10]

    # Since the inference API does not return the image path, I assume the predictions are provided in the same order as the frames in the video.
    assert len(image_paths) == len(
        predictions
    ), f"number of images must be equal to number of predictions. image_paths: {len(image_paths)} , predictions: {len(predictions)}"
    new_predictions = []

    num_kpts = len(bodyparts)

    if not superanimal_name.startswith("superanimal_"):
        raise ValueError("not supporting non superanimal model video adaptation yet")

    category_name = superanimal_name[len("superanimal_"):]
    categories = [
        {
            "name": category_name,
            "id": 1,
            "supercategory": "animal",
            "keypoints": bodyparts,
        }
    ]

    assert len(predictions) == len(image_paths)
    imageid2annotations = defaultdict(list)
    for image_id, (prediction, image_path) in enumerate(zip(predictions, image_paths)):
        image_obj = cv2.imread(image_path)
        height, width, channels = image_obj.shape
        imagename = image_path.split(os.sep)[-1]
        image = {
            "id": image_id,
            "file_name": imagename,
            "width": width,
            "height": height,
        }

        # iterate through individuals if there are many

        assert (
            len(prediction["bodyparts"])
            == len(prediction["bboxes"])
            == len(prediction["bbox_scores"])
        )
        for pose, bbox, bbox_score in zip(
            prediction["bodyparts"], prediction["bboxes"], prediction["bbox_scores"]
        ):
            if (
                np.all(np.array(pose) <= 0)
                or len(bbox) == 0
                or bbox_score < bbox_threshold
            ):
                continue
            imageid2annotations[image_id].append(pose)
            pose = np.array(pose)
            bbox = np.array(bbox)

            mask = pose[:, -1] < pose_threshold

            pose[mask] = 0

            # by default all visible
            pose[:, -1] = 2
            bbox_confidence = bbox[-1]

            keypoints = list(pose.reshape(-1))
            keypoints = [float(num) for num in keypoints]
            # bbox here is x,y,w,h from dlc3
            bbox = [float(num) for num in bbox][:4]

            anno = {
                "category_id": int(category_id),
                "keypoints": keypoints,
                "num_keypoints": len(keypoints) // 3,
                "image_id": int(image_id),
                "bbox": bbox,
                "area": float(bbox[-2] * bbox[-3]),
                "iscrowd": 0,
                "id": int(annotation_id),
            }

            annotation_id += 1
            annotations.append(anno)

        # this is to prevent images that do not have annotations
        if len(imageid2annotations[image_id]) > 0:
            images.append(image)

    train_obj = {"images": images, "annotations": annotations, "categories": categories}

    test_annotations = []

    # just use the first 10 image annotations for test
    test_obj = {
        "images": images[:10],
        "annotations": annotations[:10],
        "categories": categories,
    }

    # there is no 'test' split of video adaptation. This is essentially train.json
    with open(os.path.join(dest_proj_folder, "annotations", "test.json"), "w") as f:
        json.dump(test_obj, f, indent=4)

    with open(os.path.join(dest_proj_folder, "annotations", "train.json"), "w") as f:
        json.dump(train_obj, f, indent=4)
