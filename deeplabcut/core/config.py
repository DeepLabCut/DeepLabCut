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

import os
import warnings
from pathlib import Path
from typing import Callable

import yaml
import ruamel.yaml.representer
from ruamel.yaml import YAML

from deeplabcut.core.engine import Engine


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes

    Raises:
        FileNotFoundError: if the config file does not exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config {config_path} is not found. Please make sure that the file exists."
        )
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

# -----------------------------------------------------------------------------
# Project config (config.yaml with template and defaults)
# -----------------------------------------------------------------------------


def create_config_template(multianimal: bool = False) -> tuple:
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.

    Returns:
        (cfg_file, ruamelFile) for further editing and dumping.
    """
    if multianimal:
        yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
multianimalproject:
identity:
\n
# Project path (change when moving around)
project_path:
\n
# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch
\n
# Annotation data set configuration (and individual video cropping parameters)
video_sets:
individuals:
uniquebodyparts:
multianimalbodyparts:
bodyparts:
\n
# Fraction of video to start/stop when extracting frames for labeling/refinement
start:
stop:
numframes2pick:
\n
# Plotting configuration
skeleton:
skeleton_color:
pcutoff:
dotsize:
alphavalue:
colormap:
\n
# Training,Evaluation and Analysis configuration
TrainingFraction:
iteration:
default_net_type:
default_augmenter:
default_track_method:
snapshotindex:
detector_snapshotindex:
batch_size:
\n
# Cropping Parameters (for analysis and outlier frame detection)
cropping:
#if cropping is true for analysis, then set the values here:
x1:
x2:
y1:
y2:
\n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
corner2move2:
move2corner:
\n
# Conversion tables to fine-tune SuperAnimal weights
SuperAnimalConversionTables:
        """
    else:
        yaml_str = """\
# Project definitions (do not edit)
Task:
scorer:
date:
multianimalproject:
identity:
\n
# Project path (change when moving around)
project_path:
\n
# Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
engine: pytorch
\n
# Annotation data set configuration (and individual video cropping parameters)
video_sets:
bodyparts:
\n
# Fraction of video to start/stop when extracting frames for labeling/refinement
start:
stop:
numframes2pick:
\n
# Plotting configuration
skeleton:
skeleton_color:
pcutoff:
dotsize:
alphavalue:
colormap:
\n
# Training,Evaluation and Analysis configuration
TrainingFraction:
iteration:
default_net_type:
default_augmenter:
snapshotindex:
detector_snapshotindex:
batch_size:
detector_batch_size:
\n
# Cropping Parameters (for analysis and outlier frame detection)
cropping:
#if cropping is true for analysis, then set the values here:
x1:
x2:
y1:
y2:
\n
# Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
corner2move2:
move2corner:
\n
# Conversion tables to fine-tune SuperAnimal weights
SuperAnimalConversionTables:
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
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
    ruamelFile_3d = YAML()
    cfg_file_3d = ruamelFile_3d.load(yaml_str)
    return cfg_file_3d, ruamelFile_3d


def read_config(configname: str | Path) -> dict:
    """
    Reads structured config file defining a project.
    Applies default values and repairs (engine, detector_snapshotindex, project_path) and writes back if needed.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = str(Path(configname).parent.resolve())

                if cfg.get("engine") is None:
                    cfg["engine"] = Engine.TF.aliases[0]
                    write_project_config(configname, cfg)

                if cfg.get("detector_snapshotindex") is None:
                    cfg["detector_snapshotindex"] = -1

                if cfg.get("detector_batch_size") is None:
                    cfg["detector_batch_size"] = 1

                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_project_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_project_config(configname, cfg)
                else:
                    raise
    else:
        raise FileNotFoundError(
            f"Config file at {path} not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg


def write_project_config(configname: str | Path, cfg: dict) -> None:
    """Write structured project config file (config.yaml) preserving template order."""
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
