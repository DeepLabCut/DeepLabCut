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
"""Centralized helpers for reading, writing, and creating configuration files (YAML)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from pathlib import PurePath
from enum import Enum

if TYPE_CHECKING:
    from deeplabcut.core.config.project_config import ProjectConfig, ProjectConfig3D

import yaml
import ruamel.yaml.representer
from ruamel.yaml import YAML
from omegaconf import DictConfig, ListConfig
from pydantic import ValidationError

from deeplabcut.core.engine import Engine


def get_yaml_loader() -> YAML:
    """Get a ruamel.yaml YAML handler with safe mode."""
    yaml = YAML(typ="safe", pure=True)
    return yaml


def get_yaml_dumper() -> YAML:
    """Get a ruamel.yaml YAML handler with representers for Enum and Path objects."""
    yaml = YAML(typ="rt", pure=True)
    # Auto-serialize Path objects as strings
    yaml.representer.add_multi_representer(
        PurePath,
        lambda r, p: r.represent_str(str(p))
    )
    yaml.representer.add_multi_representer(
        Enum,
        lambda r, e: r.represent_str(e.value)
    )
    # OmegaConf containers -> plain dict/list so ruamel can serialize them
    yaml.representer.add_representer(
        DictConfig,
        lambda r, d: r.represent_dict(dict(d))
    )
    yaml.representer.add_representer(
        ListConfig,
        lambda r, l: r.represent_list(list(l))
    )
    return yaml


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes

    Raises:
        FileNotFoundError: if the config file does not exist
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"Config {config_path} is not found. Please make sure that the file exists."
        )
    with open(config_path, "r") as f:
        cfg = get_yaml_loader().load(f)

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
        get_yaml_dumper().dump(config, file)


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

# -----------------------------------------------------------------------------
# Project config (config.yaml with template and defaults)
# -----------------------------------------------------------------------------


def create_config_template(multianimal: bool = False) -> tuple:
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.

    Returns:
        (cfg_file, ruamelFile) for further editing and dumping.
    """
    warnings.warn("This function is deprecated. Use deeplabcut.core.config.ProjectConfig instead.")
    from deeplabcut.core.config.project_config import ProjectConfig
    ruamelFile = get_yaml_dumper()
    cfg_file = ProjectConfig(multianimalproject=multianimal).to_dict()
    return cfg_file, ruamelFile


def create_config_template_3d() -> tuple:
    """
    Creates a template for config.yaml file for 3d project. This specific order is preserved while saving as yaml file.

    Returns:
        (cfg_file_3d, ruamelFile_3d) for further editing and dumping.
    """
    yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
\n
# Project path (change when moving around)
project_path:
\n
# Plotting configuration
skeleton: # Note that the pairs must be defined, as you want them linked!
skeleton_color:
pcutoff:
colormap:
dotsize:
alphaValue:
markerType:
markerColor:
\n
# Number of cameras, camera names, path of the config files, shuffle index and trainingsetindex used to analyze videos:
num_cameras:
camera_names:
scorername_3d: # Enter the scorer name for the 3D output
    """
    ruamelFile_3d = get_yaml_dumper()
    cfg_file_3d = ruamelFile_3d.load(yaml_str)
    return cfg_file_3d, ruamelFile_3d


def read_config(configname: str | Path, ignore_empty: bool = True) -> DictConfig:
    """
    Reads structured config file defining a project.

    Applies default values and repairs (engine, detector_snapshotindex, project_path)
    and writes back if needed.

    Args:
        configname: Path to the project configuration file (config.yaml).
        ignore_empty: If True, empty/None values in the YAML are ignored and
            dataclass defaults are used instead. If False, empty values represent None.
            Defaults to True.

    Returns:
        The project configuration as a DictConfig.
    """
    # NOTE @deruyter92 2026-02-05: Default ignore_empty is now set to True to match
    # the prior behaviour of read_config. We should consider changing this to False
    # for stricter validation.
    from deeplabcut.core.config.project_config import ProjectConfig
    path = Path(configname)
    project_config = ProjectConfig.from_yaml(path, ignore_empty=ignore_empty)
    
    # NOTE @deruyter92 2026-02-02: copied old behaviour of writing the config back to the file.
    # We should consider separating the writing and reading instead of having inplace edits during reading.
    curr_dir = str(Path(configname).parent.resolve())
    if project_config.project_path != curr_dir:
        project_config.project_path = curr_dir
        project_config.to_yaml(configname)
    return project_config.to_dictconfig()


def write_project_config(
    configname: str | Path,
    cfg: dict | ProjectConfig | DictConfig,
) -> None:
    """Write structured project config file (config.yaml) preserving template order."""
    from deeplabcut.core.config.project_config import ProjectConfig

    try:
        project_config: ProjectConfig = ProjectConfig.from_any(cfg)
        project_config.to_yaml(configname)
        return
    except ValidationError as e:
        warnings.warn(
            f"Invalid configuration! Validation error in config file {cfg}. Error: {e}"
            "Reverting to legacy config file writing."
        )
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(
            cfg.get("multianimalproject", False)
        )
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if "skeleton" not in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)


def edit_config(
    configname: str | Path, edits: dict, output_name: str | Path = ""
) -> dict:
    """
    Convenience function to edit and save a config file from a dictionary.

    Parameters
    ----------
    configname : string
        String containing the full path of the config file in the project.
    edits : dict
        Key–value pairs to edit in config
    output_name : string, optional (default='')
        Overwrite the original config.yaml by default.
        If passed in though, new filename of the edited config.

    Examples
    --------
    config_path = 'my_stellar_lab/dlc/config.yaml'

    edits = {'numframes2pick': 5,
             'trainingFraction': [0.5, 0.8],
             'skeleton': [['a', 'b'], ['b', 'c']]}

    deeplabcut.core.config.edit_config(config_path, edits)
    """
    cfg = read_config_as_dict(configname)
    for key, value in edits.items():
        cfg[key] = value
    if not output_name:
        output_name = configname
    try:
        write_config(output_name, cfg)
    except ruamel.yaml.representer.RepresenterError:
        warnings.warn(
            "Some edits could not be written. "
            "The configuration file will be left unchanged."
        )
        for key in edits:
            cfg.pop(key)
        write_config(output_name, cfg)
    return cfg


def write_config_3d(configname: str | Path, cfg: dict) -> None:
    """Write structured 3D project config file."""
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template_3d()
        for key in cfg.keys():
            cfg_file[key] = cfg[key]
        ruamelFile.dump(cfg_file, cf)


def write_config_3d_template(
    projconfigfile: str | Path, cfg_file_3d: dict, ruamelFile_3d: YAML
) -> None:
    """Write 3D config from pre-built template and YAML instance."""
    with open(projconfigfile, "w") as cf:
        ruamelFile_3d.dump(cfg_file_3d, cf)
