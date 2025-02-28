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
from typing import Callable

from ruamel.yaml import YAML


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


def write_config(config_path: str | Path, config: dict, overwrite: bool = True) -> None:
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

    with open(config_path, "w") as file:
        YAML().dump(config, file)


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
