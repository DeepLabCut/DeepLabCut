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
"""Util functions to create pytorch pose configuration files"""
from __future__ import annotations

import copy
from pathlib import Path

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.utils import auxiliaryfunctions


def replace_default_values(
    config: dict | list,
    num_bodyparts: int | None = None,
    num_individuals: int | None = None,
    backbone_output_channels: int | None = None,
    **kwargs,
) -> dict:
    """Replaces placeholder values in a model configuration with their actual values.

    This method allows to create template PyTorch configurations for models with values
    such as "num_bodyparts", which are replaced with the number of bodyparts for a
    project when making its Pytorch configuration.

    This code can also do some basic arithmetic. You can write "num_bodyparts x 2" (or
    any factor other than 2) for location refinement channels, and the number of
    channels will be twice the number of bodyparts. You can write
    "backbone_output_channels // 2" for the number of channels in a layer, and it will
    be half the number of channels output by the backbone. You can write
    "num_bodyparts + 1" (such as for DEKR heatmaps, where a "center" bodypart is added).

    The three base placeholder values that can be computed are "num_bodyparts",
    "num_individuals" and "backbone_output_channels". You can add more through the
    keyword arguments (such as "paf_graph": list[tuple[int, int]] or
    "paf_edges_to_keep": list[int] for DLCRNet models).

    Args:
        config: the configuration in which to replace default values
        num_bodyparts: the number of bodyparts
        num_individuals: the number of individuals
        backbone_output_channels: the number of backbone output channels
        kwargs: other placeholder values to fill in

    Returns:
        the configuration with placeholder values replaced

    Raises:
        ValueError: if there is a placeholder value who's "updated" value was not
            given to the method
    """

    def get_updated_value(variable: str) -> int | list[int]:
        var_parts = variable.strip().split(" ")
        var_name = var_parts[0]
        if updated_values[var_name] is None:
            raise ValueError(
                f"Found {variable} in the configuration file, but there is no default "
                f"value for this variable."
            )

        if len(var_parts) == 1:
            return updated_values[var_name]
        elif len(var_parts) == 3:
            operator, factor = var_parts[1], var_parts[2]
            if not factor.isdigit():
                raise ValueError(f"F must be an integer in variable: {variable}")

            factor = int(factor)
            if operator == "+":
                return updated_values[var_name] + factor
            elif operator == "x":
                return updated_values[var_name] * factor
            elif operator == "//":
                return updated_values[var_name] // factor
            else:
                raise ValueError(f"Unknown operator for variable: {variable}")

        raise ValueError(
            f"Found {variable} in the configuration file, but cannot parse it."
        )

    updated_values = {
        "num_bodyparts": num_bodyparts,
        "num_individuals": num_individuals,
        "backbone_output_channels": backbone_output_channels,
        **kwargs,
    }

    config = copy.deepcopy(config)
    if isinstance(config, dict):
        keys_to_update = list(config.keys())
    elif isinstance(config, list):
        keys_to_update = range(len(config))
    else:
        raise ValueError(f"Config to update must be dict or list, found {type(config)}")

    for k in keys_to_update:
        if isinstance(config[k], (list, dict)):
            config[k] = replace_default_values(
                config[k],
                num_bodyparts,
                num_individuals,
                backbone_output_channels,
                **kwargs,
            )
        elif (
            isinstance(config[k], str)
            and config[k].strip().split(" ")[0] in updated_values.keys()
        ):
            config[k] = get_updated_value(config[k])

    return config


def update_config(config: dict, updates: dict, copy_original: bool = True) -> dict:
    """Updates items in the configuration file

    The configuration dict should only be composed of primitive Python types
    (dict, list and values). This is the case when reading the file using
    `read_config_as_dict`.

    Args:
        config: the configuration dict to update
        updates: the updates to make to the configuration dict
        copy_original: whether to copy the original dict before updating it

    Returns:
        the updated dictionary
    """
    if copy_original:
        config = copy.deepcopy(config)

    for k, v in updates.items():
        if k in config and isinstance(config[k], dict) and isinstance(v, dict):
            if k in ("optimizer", "scheduler") and config["type"] != v["type"]:
                # if changing the optimizer or scheduler type, update all values
                config[k] = v
            else:
                config[k] = update_config(config[k], v, copy_original=False)
        else:
            config[k] = copy.deepcopy(v)
    return config


def update_config_by_dotpath(
    config: dict, updates: dict, copy_original: bool = True
) -> dict:
    """Updates items in the configuration file using dot notation for nested keys

    The configuration dict should only be composed of primitive Python types
    (dict, list and values). This is the case when reading the file using
    `read_config_as_dict`.

    Args:
        config: the configuration dict to update
        updates: single-level dict with dot notation keys indicating nested paths
            e.g. {"device": "cuda", "runner.gpus": [0,1]}
        copy_original: whether to copy the original dict before updating it

    Returns:
        the updated dictionary
    """
    if copy_original:
        config = copy.deepcopy(config)

    for key, value in updates.items():
        # Split key into parts by dots
        parts = key.split(".")

        # Handle non-nested case
        if len(parts) == 1:
            config[key] = copy.deepcopy(value)
            continue

        # Navigate to nested location
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value at final location
        current[parts[-1]] = copy.deepcopy(value)

    return config


def get_config_folder_path() -> Path:
    """Returns: the Path to the folder containing the "configs" for DeepLabCut 3.0"""
    dlc_parent_path = Path(auxiliaryfunctions.get_deeplabcut_path())
    return dlc_parent_path / "pose_estimation_pytorch" / "config"


def load_base_config(config_folder_path: Path) -> dict:
    """Returns: the base configuration for all PyTorch DeepLabCut models"""
    base_dir = config_folder_path / "base"
    base_config = read_config_as_dict(base_dir / "base.yaml")
    return base_config


def load_backbones(configs_dir: Path) -> list[str]:
    """
    Args:
        configs_dir: the Path to the folder containing the "configs" for PyTorch
            DeepLabCut

    Returns:
        all backbones with default configurations that can be used
    """
    backbone_dir = configs_dir / "backbones"
    backbones = [p.stem for p in backbone_dir.iterdir() if p.suffix == ".yaml"]
    return backbones


def load_detectors(configs_dir: Path) -> list[str]:
    """
    Args:
        configs_dir: the Path to the folder containing the "configs" for PyTorch
            DeepLabCut

    Returns:
        all detectors that are available
    """
    detector_dir = configs_dir / "detectors"
    detectors = [p.stem for p in detector_dir.iterdir() if p.suffix == ".yaml"]
    return detectors


def available_models() -> list[str]:
    """Returns: the possible variants of models that can be used"""
    configs_folder_path = get_config_folder_path()
    backbones = load_backbones(configs_folder_path)
    models = set()
    for backbone in backbones:
        models.add(backbone)
        models.add("top_down_" + backbone)

    other_architectures = [
        p
        for p in configs_folder_path.iterdir()
        if p.is_dir() and not p.name in ("backbones", "base", "detectors")
    ]
    for folder in other_architectures:
        variants = [p.stem for p in folder.iterdir() if p.suffix == ".yaml"]
        for variant in variants:
            models.add(variant)

    return list(sorted(models))


def is_model_top_down(net_type: str) -> bool:
    """Checks whenever a given net_type is top-down or not"""
    if net_type not in available_models():
        raise ValueError(
            f"Model {net_type} is not part of available models, which are {str(available_models())}"
        )

    configs_dir = get_config_folder_path()
    backbones = load_backbones(configs_dir)

    if net_type.startswith("top_down_"):
        return True
    elif net_type in backbones:
        return False

    configs_dir = get_config_folder_path()

    architecture = net_type.split("_")[0]

    cfg_path = configs_dir / architecture / f"{net_type}.yaml"
    model_cfg = read_config_as_dict(cfg_path)

    return model_cfg.get("method", "BU").upper() == "TD"


def is_model_cond_top_down(net_type: str) -> bool:
    """Checks whether a given net_type is conditional top-down or not"""
    if net_type not in available_models():
        raise ValueError(
            f"Model {net_type} is not part of available models, which are {str(available_models())}"
        )

    if net_type.startswith("ctd_"):
        return True
    else:
        return False


def available_detectors() -> list[str]:
    """Returns: all the possible detectors that can be used"""
    return load_detectors(get_config_folder_path())
