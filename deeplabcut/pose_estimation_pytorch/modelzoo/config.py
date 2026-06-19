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
"""Methods to create the configuration files to fine-tune SuperAnimal models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.runners.inference import InferenceConfig

from ruamel.yaml import YAML

import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.config import (
    read_config_as_dict,
)
from deeplabcut.core.deprecation import deprecated
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.config.metadata import PoseMetadata
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_super_animal_model_config_path,
    get_super_animal_project_config_path,
)
from deeplabcut.pose_estimation_pytorch.task import Task


def build_superanimal_metadata(super_animal: str, model_name: str, max_individuals: int) -> dict:
    project_cfg_path = get_super_animal_project_config_path(super_animal=super_animal)
    project_config = read_config_as_dict(project_cfg_path)
    model_cfg_path = get_super_animal_model_config_path(model_name=model_name)
    metadata = {
        "project_path": project_config.get("project_path"),
        "pose_config_path": model_cfg_path,
        "bodyparts": af.get_bodyparts(project_config),
        "unique_bodyparts": af.get_unique_bodyparts(project_config),
        "individuals": [f"animal{i}" for i in range(max_individuals)],
        "with_identity": project_config.get("identity", False),
    }
    return metadata


def build_superanimal_inference_config(
    super_animal: str,
    model_name: str,
    detector_name: str | None,
    metadata: PoseMetadata,
    device: str | None = None,
) -> dict:

    max_individuals = metadata.num_individuals

    model_cfg_path = get_super_animal_model_config_path(model_name=model_name)
    model_config = read_config_as_dict(model_cfg_path)
    model_config = config_utils.replace_default_values(
        model_config,
        num_bodyparts=metadata.num_bodyparts,
        num_individuals=max_individuals,
        backbone_output_channels=model_config["model"]["backbone_output_channels"],
    )
    if detector_name is None and super_animal != "superanimal_humanbody":
        model_config["method"] = Task.BOTTOM_UP.aliases[0].lower()
    else:
        model_config["method"] = Task.TOP_DOWN.aliases[0].lower()
        if super_animal != "superanimal_humanbody":
            detector_cfg_path = get_super_animal_model_config_path(model_name=detector_name)
            detector_cfg = read_config_as_dict(detector_cfg_path)
            model_config["detector"] = detector_cfg
        else:
            model_config.pop("detector", None)

    model_config["metadata"] = metadata.to_dict()
    if device is not None:
        model_config["device"] = device
        if model_config.get("detector", None) is not None:
            model_config["detector"]["device"] = device
    return model_config


def build_superanimal_finetune_config(
    weight_init: WeightInitialization,
    metadata: PoseMetadata,
    model_name: str,
    detector_name: str | None,
    inference_config: InferenceConfig,
) -> dict:

    # Bodyparts mapping
    converted_bodyparts = metadata.bodyparts
    if weight_init.bodyparts is not None:
        assert len(weight_init.bodyparts) == len(weight_init.conversion_array)
        converted_bodyparts = weight_init.bodyparts
    elif len(metadata.bodyparts) != len(weight_init.conversion_array):
        raise ValueError(
            "You don't have the same number of bodyparts in your project config as "
            f"number of entries your conversion array ({metadata.bodyparts} vs "
            f"{weight_init.conversion_array}). If you're fine-tuning from "
            "SuperAnimal on a subset of your bodyparts, you must specify which "
            "ones in `WeightInitialization.bodyparts`. This should be done "
            "automatically when creating the `weight_init` with "
            "`WeightInitialization.build`."
        )

    # load the model configuration
    model_cfg = read_config_as_dict(get_super_animal_model_config_path(model_name))
    if detector_name is None:
        model_cfg["method"] = Task.BOTTOM_UP.aliases[0].lower()
        # Use default bottom-up image augmentation if no detector is given (the collate
        # function might be needed).
        config_dir = config_utils.get_config_folder_path()
        aug = read_config_as_dict(config_dir / "base" / "aug_default.yaml")
        model_cfg["data"]["train"] = aug["train"]
    else:
        model_cfg["method"] = Task.TOP_DOWN.aliases[0].lower()
        model_cfg["detector"] = read_config_as_dict(get_super_animal_model_config_path(detector_name))

    # use SuperAnimal bodyparts
    if weight_init.memory_replay:
        super_animal_project_config = read_config_as_dict(get_super_animal_project_config_path(weight_init.dataset))
        converted_bodyparts = super_animal_project_config["bodyparts"]

    model_cfg["net_type"] = model_name
    model_cfg["metadata"] = metadata.to_dict()
    model_cfg["model"] = config_utils.replace_default_values(model_cfg["model"], num_bodyparts=len(converted_bodyparts))
    model_cfg["train_settings"]["weight_init"] = weight_init.to_dict()
    model_cfg["inference"] = inference_config.to_dict()
    return model_cfg


@deprecated(replacement="PoseConfig.build_for_superanimal_finetune", since="3.1")
def make_super_animal_finetune_config(
    weight_init: WeightInitialization,
    project_config: dict,
    pose_config_path: str | Path,
    model_name: str,
    detector_name: str | None,
    save: bool = False,
) -> PoseConfig:
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig

    return PoseConfig.build_for_superanimal_finetune(
        weight_init=weight_init,
        project_config=project_config,
        pose_config_path=pose_config_path,
        model_name=model_name,
        detector_name=detector_name,
        save=save,
    )


def write_pytorch_config_for_memory_replay(config_path, shuffle, pytorch_config):
    cfg = af.read_config(config_path)
    trainIndex = 0
    dlc_proj_root = Path(config_path).parent
    model_folder = dlc_proj_root / af.get_model_folder(
        cfg["TrainingFraction"][trainIndex], shuffle, cfg, engine=Engine.PYTORCH
    )
    os.makedirs(model_folder / "train", exist_ok=True)
    out_path = model_folder / "train" / "pytorch_config.yaml"
    with open(str(out_path), "w") as f:
        yaml = YAML()
        yaml.dump(pytorch_config, f)
