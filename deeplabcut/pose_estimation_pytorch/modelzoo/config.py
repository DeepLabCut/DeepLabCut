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

import deeplabcut.pose_estimation_pytorch.config.utils as config_utils
import deeplabcut.pose_estimation_pytorch.modelzoo.utils as modelzoo_utils
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.weight_init import WeightInitialization
from pathlib import Path
from deeplabcut.core.engine import Engine
from ruamel.yaml import YAML
import os

def make_super_animal_finetune_config(
    weight_init: WeightInitialization,
    project_config: dict,
    pose_config_path: str,
    net_type: str | None = None,
) -> dict:
    """
    Creates a PyTorch pose configuration file to finetune a SuperAnimal model on a
    downstream project.

    Args:
        weight_init: The weight initialization configuration.
        project_config: The project configuration.
        pose_config_path: The path where the pose configuration file will be saved
        net_type: The type of neural net to finetune.

    Returns:
        The generated pose configuration file.

    Raises:
        ValueError: If `weight_init.with_decoder = False`. This method only creates
            configs to fine-tune SuperAnimal models. Call `make_pytorch_pose_config`
            to create configuration files for transfer learning.
    """
    bodyparts = af.get_bodyparts(project_config)
    if not weight_init.with_decoder:
        raise ValueError(
            "Can call ``make_super_animal_finetune_config`` when `with_decoder=True`, "
            f" but you had {weight_init}. Please set `with_decoder=True` to fine-tune "
            "a model or call `make_pytorch_pose_config` to create a transfer learning "
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
    return create_config_from_modelzoo(
        net_type=net_type,
        super_animal=weight_init.dataset,
        converted_bodyparts=converted_bodyparts,
        weight_init=weight_init,
        project_config=project_config,
        pose_config_path=pose_config_path,
    )


def create_config_from_modelzoo(
    net_type: str,
    super_animal: str,
    converted_bodyparts: list[str],
    weight_init: WeightInitialization,
    project_config: dict,
    pose_config_path: str,
) -> dict:
    """Creates a model configuration file to fine-tune a SuperAnimal model

    Args:
        net_type: The type of neural net to finetune.
        super_animal: The SuperAnimal model to finetune.
        converted_bodyparts: The project bodyparts that the model will learn.
        weight_init: The weight initialization to use.
        project_config: The project configuration.
        pose_config_path: The path where the pose configuration file will be saved.

    Returns:
        The generated pose configuration file.
    """
    # load the SuperAnimal model config
    pose_config, project_cfg, _, _ = modelzoo_utils.get_config_model_paths(
        project_name=super_animal,
        pose_model_type=modelzoo_utils.get_pose_model_type(net_type),
    )

    # use SuperAnimal bodyparts
    if weight_init.memory_replay:
        converted_bodyparts = project_cfg["bodyparts"]

    pose_config["net_type"] = net_type
    pose_config["metadata"] = {
        "project_path": project_config["project_path"],
        "pose_config_path": pose_config_path,
        "bodyparts": converted_bodyparts,
        "unique_bodyparts": [],
        "individuals": project_config.get("individuals", ["animal"]),
        "with_identity": False,
    }

    pose_config["model"] = config_utils.replace_default_values(
        pose_config["model"], num_bodyparts=len(converted_bodyparts)
    )
    pose_config["train_settings"]["weight_init"] = weight_init.to_dict()

    # sort first-level keys to make it prettier
    return dict(sorted(pose_config.items()))

def write_pytorch_config_for_memory_replay(config_path,
                                           shuffle,
                                           pytorch_config):

    cfg = af.read_config(config_path)

    trainIndex = 0    

    dlc_proj_root = Path(config_path).parent
    
    model_folder = dlc_proj_root / af.get_model_folder(
        cfg['TrainingFraction'][trainIndex], shuffle, cfg, engine=Engine.PYTORCH)
    

    os.makedirs(model_folder / 'train', exist_ok = True)
    
    out_path = model_folder / 'train' / 'pytorch_config.yaml'


    
    with open(str(out_path), 'w') as f:
        yaml = YAML()
        yaml.dump(pytorch_config, f)
