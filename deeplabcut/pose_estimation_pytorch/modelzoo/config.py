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
    """
    bodyparts = af.get_bodyparts(project_config)
    if weight_init is not None and weight_init.with_decoder:
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
