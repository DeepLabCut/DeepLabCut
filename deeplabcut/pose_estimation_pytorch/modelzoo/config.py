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
"""Methods to create the configuration files to fine-tune SuperAnimal models"""
from __future__ import annotations

import os
from pathlib import Path

from ruamel.yaml import YAML

import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.config import (
    read_config_as_dict,
    write_config,
)
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
    get_super_animal_model_config_path,
    get_super_animal_project_config_path,
)
from deeplabcut.pose_estimation_pytorch.task import Task


def make_super_animal_finetune_config(
    weight_init: WeightInitialization,
    project_config: dict,
    pose_config_path: str | Path,
    model_name: str,
    detector_name: str | None,
    save: bool = False,
) -> dict:
    """
    Creates a PyTorch pose configuration file to finetune a SuperAnimal model on a
    downstream project.

    Args:
        weight_init: The weight initialization configuration.
        project_config: The project configuration.
        pose_config_path: The path where the pose configuration file will be saved
        model_name: The type of neural net to finetune.
        detector_name: The type of detector to use for the SuperAnimal model. If None is
            given, the model will be set to a Bottom-Up framework.
        save: Whether to save the model configuration file to the ``pose_config_path``.

    Returns:
        The generated pose configuration file.

    Raises:
        ValueError: If `weight_init.with_decoder = False`. This method only creates
            configs to fine-tune SuperAnimal models. Call `make_pytorch_pose_config`
            to create configuration files for transfer learning.
    """
    bodyparts = af.get_bodyparts(project_config)
    if weight_init.dataset is None:
        raise ValueError(
            "You must set the ``WeightInitialization.dataset`` when fine-tuning "
            "SuperAnimal models."
        )

    if not weight_init.with_decoder:
        raise ValueError(
            "Can only call ``make_super_animal_finetune_config`` when "
            f" `with_decoder=True`, but you had {weight_init}. Please set "
            "`with_decoder=True` to fine-tune a model or call "
            "`make_pytorch_pose_config` to create a transfer learning "
            "pose configuration file."
        )

    converted_bodyparts = bodyparts
    if weight_init.bodyparts is not None:
        assert len(weight_init.bodyparts) == len(weight_init.conversion_array)
        converted_bodyparts = weight_init.bodyparts
    elif len(bodyparts) != len(weight_init.conversion_array):
        raise ValueError(
            "You don't have the same number of bodyparts in your project config as "
            f"number of entries your conversion array ({bodyparts} vs "
            f"{weight_init.conversion_array}). If you're fine-tuning from "
            "SuperAnimal on a subset of your bodyparts, you must specify which "
            "ones in `WeightInitialization.bodyparts`. This should be done "
            "automatically when creating the `weight_init` with "
            "`WeightInitialization.build`."
        )

    # Load the exact pose configuration file for the model to fine-tune
    pose_config = create_config_from_modelzoo(
        super_animal=weight_init.dataset,
        model_name=model_name,
        detector_name=detector_name,
        converted_bodyparts=converted_bodyparts,
        weight_init=weight_init,
        project_config=project_config,
        pose_config_path=pose_config_path,
    )
    if save:
        write_config(pose_config_path, pose_config, overwrite=True)

    return pose_config


def create_config_from_modelzoo(
    super_animal: str,
    model_name: str,
    detector_name: str | None,
    converted_bodyparts: list[str],
    weight_init: WeightInitialization,
    project_config: dict,
    pose_config_path: str | Path,
) -> dict:
    """Creates a model configuration file to fine-tune a SuperAnimal model

    Args:
        super_animal: The SuperAnimal dataset on which the model was trained.
        model_name: The type of neural net to finetune.
        detector_name: The type of detector to use for the SuperAnimal model. If None is
            given, the model will be set to a Bottom-Up framework.
        converted_bodyparts: The project bodyparts that the model will learn.
        weight_init: The weight initialization to use.
        project_config: The project configuration.
        pose_config_path: The path where the pose configuration file will be saved.

    Returns:
        The generated pose configuration file.
    """
    # load the model configuration
    model_cfg = read_config_as_dict(
        get_super_animal_model_config_path(model_name)
    )
    if detector_name is None:
        model_cfg["method"] = Task.BOTTOM_UP.aliases[0].lower()
        # Use default bottom-up image augmentation if no detector is given (the collate
        # function might be needed).
        config_dir = config_utils.get_config_folder_path()
        aug = read_config_as_dict(config_dir / "base" / "aug_default.yaml")
        model_cfg["data"]["train"] = aug["train"]
    else:
        model_cfg["method"] = Task.TOP_DOWN.aliases[0].lower()
        model_cfg["detector"] = read_config_as_dict(
            get_super_animal_model_config_path(detector_name)
        )

    # use SuperAnimal bodyparts
    if weight_init.memory_replay:
        super_animal_project_config = read_config_as_dict(
            get_super_animal_project_config_path(super_animal)
        )
        converted_bodyparts = super_animal_project_config["bodyparts"]

    model_cfg["net_type"] = model_name
    model_cfg["metadata"] = {
        "project_path": project_config["project_path"],
        "pose_config_path": str(pose_config_path),
        "bodyparts": converted_bodyparts,
        "unique_bodyparts": [],
        "individuals": project_config.get("individuals", ["animal"]),
        "with_identity": False,
    }

    model_cfg["model"] = config_utils.replace_default_values(
        model_cfg["model"], num_bodyparts=len(converted_bodyparts)
    )
    model_cfg["train_settings"]["weight_init"] = weight_init.to_dict()

    # sort first-level keys to make it prettier
    return dict(sorted(model_cfg.items()))


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
