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
import inspect
import os
import subprocess
import warnings
from pathlib import Path

import torch
from dlclibrary import download_huggingface_model

import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import add_metadata
from deeplabcut.utils import auxiliaryfunctions


def get_config_model_paths(
    project_name: str,
    pose_model_type: str,
    detector_type: str = "fasterrcnn",
    weight_folder: str = None,
):
    """Get the paths to the model and project configs

    Args:
        project_name: the name of the project
        pose_model_type: the name of the pose model
        detector_type: the type of the detector
        weight_folder: the folder containing the weights

    Returns:
        the paths to the models and project configs
    """
    dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")

    model_cfg_path = os.path.join(
        modelzoo_path, "model_configs", f"{pose_model_type}.yaml"
    )
    model_config = auxiliaryfunctions.read_plainconfig(model_cfg_path)
    project_config = auxiliaryfunctions.read_config(
        os.path.join(modelzoo_path, "project_configs", f"{project_name}.yaml")
    )

    model_config = add_metadata(project_config, model_config, model_cfg_path)
    if weight_folder is None:
        weight_folder = os.path.join(modelzoo_path, "checkpoints")

    # FIXME - DO NOT DOWNLOAD HERE
    pose_model_name = f"{project_name}_{pose_model_type}.pth"
    pose_model_path = os.path.join(weight_folder, pose_model_name)
    detector_name = f"{project_name}_{detector_type}.pt"
    detector_model_path = os.path.join(weight_folder, detector_name)
    if not (Path(pose_model_path).exists() and Path(detector_model_path).exists()):
        download_huggingface_model(
            f"{project_name}_{pose_model_type}",
            target_dir=str(weight_folder),
            rename_mapping={
                "pose_model.pth": pose_model_name,
                "detector.pt": detector_name,
            },
        )

    # FIXME: Needed due to changes in code - remove when new snapshots are uploaded
    pose_model_path = _parse_model_snapshot(Path(pose_model_path), device="cpu")
    detector_model_path = _parse_model_snapshot(Path(detector_model_path), device="cpu")

    return (
        model_config,
        project_config,
        pose_model_path,
        detector_model_path,
    )


def get_gpu_memory_map():
    """Get the current gpu usage."""
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    return gpu_memory_map


def select_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:0")
    else:
        return torch.device("cpu")


def raise_warning_if_called_directly():
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    caller_name = caller_frame[1].filename

    if not "pose_estimation_" in caller_name:
        warnings.warn(
            f"{caller_name} is intended for internal use only and should not be called directly.",
            UserWarning,
        )


def update_config(config, max_individuals, device):
    config = config_utils.replace_default_values(
        config,
        num_bodyparts=len(config["bodyparts"]),
        num_individuals=max_individuals,
        backbone_output_channels=config["model"]["backbone_output_channels"],
    )
    config["device"] = device
    config_utils.pretty_print(config)
    return config


def _parse_model_snapshot(base: Path, device: str, print_keys: bool = False) -> Path:
    """FIXME: A new snapshot should be uploaded and used"""

    def _map_model_keys(state_dict: dict) -> dict:
        updated_dict = {}
        for k, v in state_dict.items():
            if not (
                k.startswith("backbone.model.downsamp_modules.")
                or k.startswith("backbone.model.final_layer")
                or k.startswith("backbone.model.classifier")
            ):
                parts = k.split(".")
                if parts[:4] == ["heads", "bodypart", "heatmap_head", "model"]:
                    parts[3] = "deconv_layers.0"
                updated_dict[".".join(parts)] = v
        return updated_dict

    parsed = base.with_stem(base.stem + "_parsed")
    if not parsed.exists():
        snapshot = torch.load(base, map_location=device)
        if print_keys:
            print(5 * "-----\n")
            print(base.stem + " keys")
            for name, _ in snapshot["model_state_dict"].items():
                print(f"  * {name}")
            print()

        parsed_model_snapshot = {
            "model": _map_model_keys(snapshot["model_state_dict"]),
            "metadata": {"epoch": 0},
        }
        torch.save(parsed_model_snapshot, parsed)
    return parsed


def get_pose_model_type(backbone: str) -> str:
    """Temporary fix: pose_model_types for SuperAnimal models do not match net types"""
    if backbone.startswith("resnet"):
        return backbone
    elif backbone.startswith("hrnet"):
        return backbone.replace("_", "")

    raise ValueError(f"Unknown backbone for SuperAnimal Weights")
