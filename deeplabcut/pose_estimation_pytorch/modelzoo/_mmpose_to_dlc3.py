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
import argparse
import json
import os
from pathlib import Path

import torch

from deeplabcut.pose_estimation_pytorch.apis.utils import build_pose_model
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.utils import auxiliaryfunctions


def _map_modelzoo_to_dlc(state_dict: dict) -> dict:
    """Map the model zoo weights to the DLC format
    Args:
        state_dict: the model zoo state dict
    Returns:
        the mapped state dict
    """

    updated_dict = {}
    for k, v in state_dict.items():
        parts = k.split(".")
        if parts[0] == "backbone":
            parts = [parts[0], "model", *parts[1:]]
        elif parts[0] == "keypoint_head":
            parts = ["heads", "bodypart", "heatmap_head", "model", parts[-1]]
            if parts[-1] == "weight":
                v = v.permute(1, 0, 2, 3)

        updated_dict[".".join(parts)] = v
    return updated_dict


def map_modelzoo_to_dlc(
    model_zoo_weights_path: str, device: str, pytorch_config: dict
) -> PoseModel:
    """Map the model zoo weights to the DLC format
    Args:
        model_zoo_weights_path: the path to the model zoo weights
        device: the device to load the weights on
        pytorch_config: the pytorch config to use for model building
    Returns:
        the mapped state dict
    """
    model_weights = torch.load(
        str(model_zoo_weights_path), map_location=torch.device(device)
    )
    model = build_pose_model(pytorch_config)
    mising_keys = model.load_state_dict(
        _map_modelzoo_to_dlc(model_weights["state_dict"]), strict=False
    )

    assert len(mising_keys[1]) == 0

    return model


def _shift_category_ids(ann_files):
    """Shift category ids to 1"""

    for mode in ["train", "test"]:
        with open(ann_files[mode], "r") as f:
            data = json.load(f)

        cleaned_data = data.copy()
        for cat in cleaned_data["categories"]:
            cat["id"] = 1

        for ann in cleaned_data["annotations"]:
            ann["category_id"] = 1

        with open(ann_files[mode], "w") as f:
            json.dump(cleaned_data, f)


def _clean_result_json(results_files):
    """Clean the json files"""

    for mode in ["train", "test"]:
        with open(results_files[mode], "r") as f:
            data = json.load(f)

        out_anns = []
        for ann in data:
            ann["score"] = ann["bbox_scores"][0]
            out_anns.append(ann)

        with open(results_files[mode], "w") as f:
            json.dump(out_anns, f)


def _change_path_annotations(ann_files, project_root):
    for mode in ["train", "test"]:
        with open(ann_files[mode], "r") as f:
            data = json.load(f)
        for i in range(len(data["images"])):
            basename = os.path.basename(data["images"][i]["file_name"])
            data["images"][i]["file_name"] = f"{project_root}/images/{basename}"


def modify_annotations(project_root, ann_files):
    _shift_category_ids(ann_files)
    _change_path_annotations(ann_files, project_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("pytorch_config_path")
    parser.add_argument("model_zoo_weights_path")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    pytorch_config = auxiliaryfunctions.read_config(args.pytorch_config_path)
    backbone_type = pytorch_config["model"]["backbone"]["model_name"]
    save_path = (
        f"{os.path.dirname(args.pytorch_config_path)}/checkpoints/{backbone_type}.pth"
    )
    annotatin_files = {
        "train": Path(args.project_root) / "annotations" / args.test_file,
        "test": Path(args.project_root) / "annotations" / args.train_file,
    }

    modify_annotations(args.project_root, annotatin_files)
    model = map_modelzoo_to_dlc(
        args.model_zoo_weights_path, args.device, pytorch_config
    )

    torch.save(model.state_dict(), save_path)
