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
import subprocess
import warnings
from pathlib import Path

import torch
from dlclibrary import download_huggingface_model
import huggingface_hub

import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import add_metadata
from deeplabcut.utils import auxiliaryfunctions


def get_model_configs_folder_path() -> Path:
    """Returns: the folder containing the SuperAnimal model configuration files"""
    return Path(auxiliaryfunctions.get_deeplabcut_path()) / "modelzoo" / "model_configs"


def get_project_configs_folder_path() -> Path:
    """Returns: the folder containing the SuperAnimal project configuration files"""
    return (
        Path(auxiliaryfunctions.get_deeplabcut_path()) / "modelzoo" / "project_configs"
    )


def get_snapshot_folder_path() -> Path:
    """Returns: the path to the folder containing the SuperAnimal model snapshots"""
    return Path(auxiliaryfunctions.get_deeplabcut_path()) / "modelzoo" / "checkpoints"


def get_super_animal_model_config_path(model_name: str, super_animal: str = None) -> Path:
    """Gets the path to the configuration file for a SuperAnimal model.

    Args:
        model_name: The name of the model for which to get the path.
        super_animal: The name of the SuperAnimal (used for specific model configs).

    Returns:
        The path to the config file for a SuperAnimal model.
    """
    # Special case for superanimal_humanbody with rtmpose_x
    if model_name == "rtmpose_x" and super_animal == "superanimal_humanbody":
        return get_model_configs_folder_path() / "superanimal_humanbody_rtmpose_x.yaml"
    
    return get_model_configs_folder_path() / f"{model_name}.yaml"


def get_super_animal_project_config_path(super_animal: str) -> Path:
    """Gets the path to a SuperAnimal project configuration file.

    Args:
        super_animal: The name of the SuperAnimal for which to get the config path.

    Returns:
        The path to the config file for a SuperAnimal project.
    """
    return get_project_configs_folder_path() / f"{super_animal}.yaml"


def get_super_animal_snapshot_path(
    dataset: str,
    model_name: str,
    download: bool = True,
) -> Path:
    """Gets the path to the snapshot containing SuperAnimal model weights.

    Args:
        dataset: The name of the SuperAnimal dataset.
        model_name: The name of the model.
        download: Whether to download the weights if they aren't already there.

    Returns:
        The path to the weights for a SuperAnimal model.
    """
    model_path = get_snapshot_folder_path() / f"{dataset}_{model_name}.pt"
    if download and not model_path.exists():
        download_super_animal_snapshot(dataset, model_name)

    return model_path


def load_super_animal_config(
    super_animal: str,
    model_name: str,
    detector_name: str | None = None,
    max_individuals: int = 30,
    device: str | None = None,
) -> dict:
    """Loads the model configuration file for a model, detector and SuperAnimal

    Args:
        super_animal: The name of the SuperAnimal for which to create the model config.
        model_name: The name of the model for which to create the model config.
        detector_name: The name of the detector for which to create the model config.
        max_individuals: The maximum number of detections to make in an image
        device: The device to use to train/run inference on the model

    Returns:
        The model configuration for a SuperAnimal-pretrained model.
    """
    project_cfg_path = get_super_animal_project_config_path(super_animal=super_animal)
    project_config = read_config_as_dict(project_cfg_path)

    # Special handling for superanimal_humanbody with rtmpose_x - download config from HuggingFace
    if super_animal == "superanimal_humanbody" and model_name == "rtmpose_x":
        # Download config from HuggingFace
        model_files = get_snapshot_folder_path()
        model_files.mkdir(exist_ok=True)
        
        path_model_config = Path(
            huggingface_hub.hf_hub_download(
                "DeepLabCut/HumanBody",
                "rtmpose-x_simcc-body7_pytorch_config.yaml",
                local_dir=model_files,
            )
        )
        model_config = read_config_as_dict(path_model_config)
    else:
        # Use local config file for other models
        model_cfg_path = get_super_animal_model_config_path(model_name=model_name, super_animal=super_animal)
        model_config = read_config_as_dict(model_cfg_path)
    
    model_config = add_metadata(project_config, model_config, model_cfg_path if 'model_cfg_path' in locals() else path_model_config)

    if detector_name is None:
        model_config["method"] = "BU"
    else:
        # Check if this is a torchvision detector (not in dlclibrary)
        if super_animal == "superanimal_humanbody" and detector_name == "fasterrcnn_mobilenet_v3_large_fpn":
            # Use torchvision detector - set method to TD and load detector config
            model_config["method"] = "TD"
            detector_cfg_path = get_super_animal_model_config_path(model_name=detector_name, super_animal=super_animal)
            detector_cfg = read_config_as_dict(detector_cfg_path)
            model_config["detector"] = detector_cfg
        else:
            # Load detector config from dlclibrary
            detector_cfg_path = get_super_animal_model_config_path(model_name=detector_name, super_animal=super_animal)
            detector_cfg = read_config_as_dict(detector_cfg_path)
            model_config["method"] = "TD"
            model_config["detector"] = detector_cfg
    
    # Update config after detector is added (if any)
    model_config = update_config(model_config, max_individuals, device)
    
    # Add superanimal_name to metadata for all superanimal models (needed for detector routing)
    if "metadata" not in model_config:
        model_config["metadata"] = {}
    model_config["metadata"]["superanimal_name"] = super_animal
    
    return model_config


def download_super_animal_snapshot(dataset: str, model_name: str) -> Path:
    """Downloads a SuperAnimal snapshot

    Args:
        dataset: The name of the SuperAnimal dataset for which to download a snapshot.
        model_name: The name of the model for which to download a snapshot.

    Returns:
        The path to the downloaded snapshot.

    Raises:
        RuntimeError if the model fails to download.
    """
    snapshot_dir = get_snapshot_folder_path()
    full_model_name = f"{dataset}_{model_name}"
    model_path = snapshot_dir / f"{full_model_name}.pt"

    # Use the full name for dlclibrary lookup (consistent with dlclibrary naming)
    download_huggingface_model(full_model_name, target_dir=str(snapshot_dir))
    
    # Check if the file was downloaded with the expected name
    if not model_path.exists():
        # If not, look for the actual downloaded filename and rename it
        if dataset == "superanimal_humanbody" and model_name == "rtmpose_x":
            actual_file = snapshot_dir / "rtmpose-x_simcc-body7.pt"
            if actual_file.exists():
                actual_file.rename(model_path)
            else:
                raise RuntimeError(f"Failed to download {model_name} to {model_path}")
        else:
            raise RuntimeError(f"Failed to download {model_name} to {model_path}")

    return snapshot_dir / f"{full_model_name}.pt"


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


def update_config(config: dict, max_individuals: int, device: str):
    """Loads the model configuration file for a model, detector and SuperAnimal

    Args:
        config: The default model configuration file.
        max_individuals: The maximum number of detections to make in an image
        device: The device to use to train/run inference on the model

    Returns:
        The model configuration for a SuperAnimal-pretrained model.
    """
 
    
    config = config_utils.replace_default_values(
        config,
        num_bodyparts=len(config["metadata"]["bodyparts"]),
        num_individuals=max_individuals,
        backbone_output_channels=config["model"]["backbone_output_channels"],
    )
    config["metadata"]["individuals"] = [f"animal{i}" for i in range(max_individuals)]

    config["device"] = device
    if "detector" in config:
        config["detector"]["device"] = device
    return config
