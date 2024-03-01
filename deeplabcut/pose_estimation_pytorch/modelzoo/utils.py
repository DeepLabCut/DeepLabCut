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
import json
import os
import subprocess
import warnings

import torch

from deeplabcut.utils import auxiliaryfunctions


def _get_config_model_paths(
    project_name: str,
    pose_model_type: str,
    detector_type: str = "fasterrcnn",
    weight_folder: str = None,
):
    """Get the paths to the model and project configs

    Args:
        project_name: the name of the project
        pose_model_name: the name of the pose model
        detector_type: the type of the detector
        weight_folder: the folder containing the weights
    Returns:
        the paths to the models and project configs
    """
    dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")

    model_config = auxiliaryfunctions.read_plainconfig(
        os.path.join(modelzoo_path, "model_configs", f"{pose_model_type}.yaml")
    )
    project_config = auxiliaryfunctions.read_config(
        os.path.join(modelzoo_path, "project_configs", f"{project_name}.yaml")
    )
    if weight_folder is None:
        weight_folder = os.path.join(modelzoo_path, "checkpoints")

    pose_model_path = os.path.join(
        weight_folder, f"{project_name}_{pose_model_type}.pth"
    )
    detector_model_path = os.path.join(
        weight_folder, f"{project_name}_{detector_type}.pt"
    )

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
        gpu_memory_map = get_gpu_memory_map()
        selected_device = max(gpu_memory_map, key=gpu_memory_map.get)
        print(f"Device was set to cuda:{selected_device}")

        return torch.device(f"cuda:{selected_device}")
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


def _update_config(config, max_individuals, device):
    print(config)
    num_bodyparts = len(config["bodyparts"])
    config["detector"]["runner"]["max_individuals"] = max_individuals
    config["multianimalproject"] = max_individuals > 1
    config["individuals"] = ["animal"]
    config["multianimalbodyparts"] = config["bodyparts"]
    config["uniquebodyparts"] = []
    config["device"] = device
    config["model"]["heads"]["bodypart"]["target_generator"][
        "num_heatmaps"
    ] = num_bodyparts
    config["model"]["heads"]["bodypart"]["heatmap_config"]["channels"][
        -1
    ] = num_bodyparts
    config["individuals"] = ["single"] * max_individuals

    return config
