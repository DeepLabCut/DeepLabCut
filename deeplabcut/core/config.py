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
"""Simple helper methods related to configuration files stored in yaml files"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TYPE_CHECKING

from ruamel.yaml import YAML
from omegaconf import OmegaConf, DictConfig
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig


def dict_to_pose_config(cfg_dict: dict) -> "PoseConfig":
    """
    Args:
        config_dict: the configuration as a dictionary

    Returns:
        The configuration file as a DictConfig
    """
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    
    cfg_dictconf = OmegaConf.create(cfg_dict)
    # Validate the config
    TypeAdapter(PoseConfig).validate_python(
        OmegaConf.to_container(cfg_dictconf, resolve=True),
        extra="forbid",
    )

    return cfg_dictconf


def read_config_as_pose_config(config_path: str | Path) -> "PoseConfig":
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes
    """
    return dict_to_pose_config(read_config_as_dict(config_path))


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes
    """
    with open(config_path, "r") as f:
        cfg = YAML(typ="safe", pure=True).load(f)

    return cfg


def load_config(config: "PoseConfig | dict | str | Path") -> "PoseConfig | DictConfig":
    """
    Loads the pose configuration from a file path or dictionary.
    Args:
        config: The pose configuration as a PoseConfig, dictionary, or path to a file.

    Returns:
        The pose configuration as a PoseConfig or DictConfig.
    """
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    
    if isinstance(config, (str, Path)):
        config = read_config_as_pose_config(config)
    elif isinstance(config, dict):
        config = dict_to_pose_config(config)
    assert isinstance(config, PoseConfig) or isinstance(config, DictConfig)
    return config


def write_config(
    config_path: str | Path, config: DictConfig, overwrite: bool = True
) -> None:
    """Writes a pose configuration file to disk

    Args:
        config_path: the path where the config should be saved
        config: the config to save
        overwrite: whether to overwrite the file if it already exists

    Raises:
        FileExistsError if overwrite=True and the file already exists
    """
    if not overwrite and Path(config_path).exists():
        raise FileExistsError(
            f"Cannot write to {config_path} - set overwrite=True to force"
        )

    OmegaConf.save(config, config_path)


def pretty_print(
    config: dict,
    indent: int = 0,
    print_fn: Callable[[str], None] | None = None,
) -> None:
    """Prints a model configuration in a pretty and readable way

    Args:
        config: the config to print
        indent: the base indent on all keys
        print_fn: custom function to call (simply calls ``print`` if None)
    """
    if print_fn is None:
        print_fn = print

    for k, v in config.items():
        if isinstance(v, dict):
            print_fn(f"{indent * ' '}{k}:")
            pretty_print(v, indent + 2, print_fn=print_fn)
        else:
            print_fn(f"{indent * ' '}{k}: {v}")
