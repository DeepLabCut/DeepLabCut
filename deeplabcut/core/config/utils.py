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
"""Centralized helpers for reading, writing, and creating configuration files (YAML).

The generic helpers are dlccore's, re-exported here.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deeplabcut.core.config.project_config import ProjectConfig
import ruamel.yaml.representer
from dlccore.config.utils import (
    get_yaml_dumper,
    get_yaml_loader,
    normalize_for_serialization,
    pretty_print,
    read_config_as_dict,
    resolve_alias,
    resolve_aliases_in_dict,
    write_config,
)
from pydantic import ValidationError
from ruamel.yaml import YAML

from deeplabcut.core.deprecation import deprecated

logger = logging.getLogger(__name__)

__all__ = [
    "create_config_template",
    "create_config_template_3d",
    "edit_config",
    "ensure_plain_config",
    "get_yaml_dumper",
    "get_yaml_loader",
    "normalize_for_serialization",
    "pretty_print",
    "read_config",
    "read_config_as_dict",
    "resolve_alias",
    "resolve_aliases_in_dict",
    "write_config",
    "write_config_3d",
    "write_config_3d_template",
    "write_project_config",
]


def ensure_plain_config(fn: Callable) -> Callable:
    """Convert typed config arguments into plain Python objects.

    Any positional or keyword argument that is a DLCBaseConfig is converted to
    a plain ``dict`` before the decorated function is called.
    """

    def _to_plain(value, fn_name: str = "<unknown>", var_name: str = "<unknown>"):
        # Lazy import to avoid circular imports during module initialization.
        from deeplabcut.core.config.base_config import DLCBaseConfig

        if isinstance(value, DLCBaseConfig):
            logger.debug(
                "converting %s (%s) to native dict in %s.",
                var_name,
                type(value).__name__,
                fn_name,
            )
            return value.to_dict()
        return value

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn_name = fn.__qualname__
        args = tuple(_to_plain(a, fn_name) for a in args)
        kwargs = {k: _to_plain(v, fn_name=fn_name, var_name=k) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


# -----------------------------------------------------------------------------
# Project config (config.yaml with template and defaults)
# -----------------------------------------------------------------------------


@deprecated(replacement="deeplabcut.core.config.ProjectConfig", since="3.0.1")
def create_config_template(multianimal: bool = False) -> tuple:
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.

    Returns:
        (cfg_file, ruamelFile) for further editing and dumping.
    """
    from deeplabcut.core.config.project_config import ProjectConfig

    ruamelFile = get_yaml_dumper()

    # TODO @deruyter92 2026-06-15: This sentinel should be removed in v1.
    bodyparts = "MULTI!" if multianimal else []
    cfg_file = ProjectConfig(multianimalproject=multianimal, bodyparts=bodyparts).to_dict()
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


_IGNORE_EMPTY_UNSET = object()


def read_config(
    configname: str | Path,
    ignore_empty: bool | object = _IGNORE_EMPTY_UNSET,
) -> ProjectConfig:
    """
    Reads structured config file defining a project.

    Applies default values and repairs (engine, detector_snapshotindex, project_path)
    and writes back if needed.

    Args:
        configname: Path to the project configuration file (config.yaml).
        ignore_empty: Deprecated, has no effect. Empty/None values for
            non-optional config.yaml fields are always treated as unset; optional
            fields retain their ``None`` values (see `ProjectConfig` validators).

    Returns:
        The project configuration as a ProjectConfig instance (supports dict-like access).
    """
    from deeplabcut.core.config.project_config import ProjectConfig
    from deeplabcut.core.deprecation import DLCDeprecationWarning

    if ignore_empty is not _IGNORE_EMPTY_UNSET:
        warnings.warn(
            "read_config(ignore_empty=...) is deprecated and no longer has any effect: "
            "empty/None values for non-optional ProjectConfig fields are always treated "
            "as unset; optional fields retain their None values.",
            DLCDeprecationWarning,
            stacklevel=2,
        )

    path = Path(configname)
    project_config = ProjectConfig.from_yaml(path)

    # If necessary, ProjectConfig automatically updates its project path via _post_yaml_load_updates.
    # if that is the case (marked as dirty), we write the config back to the file.
    if "project_path" in project_config.dirty_fields:
        # NOTE @deruyter92 2026-02-02: copied old behaviour of writing the config
        # immediately back to the file after reading it. We should consider separating
        # the writing and reading instead of having inplace edits during reading.
        project_config.to_yaml(configname, log_changes=True, mark_clean=True)
    return project_config


def write_project_config(
    configname: str | Path,
    cfg: dict | ProjectConfig,
) -> None:
    """
    Write structured project config file (config.yaml) preserving template order.

    Args:
        configname (str | Path): Path to the project configuration file (config.yaml).
        cfg (dict | ProjectConfig): The project configuration to write (requires ProjectConfig schema).

    Note:
        Validates before writing when possible, unvalidated legacy dump on failure; This may not round-trip via
        read_config for non-conformant legacy configurations.
    """
    from deeplabcut.core.config.base_config import DLCBaseConfig
    from deeplabcut.core.config.project_config import ProjectConfig

    try:
        project_config: ProjectConfig = ProjectConfig.from_any(cfg)
        project_config.to_yaml(configname)
        return
    except ValidationError as e:
        logger.error(
            "Invalid configuration! Validation error in project config file %s. Error: %s "
            "Trying to write the file to disk anyway. Please verify the config file.",
            cfg,
            e,
        )
    if isinstance(cfg, DLCBaseConfig):
        logger.error("Expected a ProjectConfig, got %s.", type(cfg).__name__)
        cfg.to_yaml(configname)
        return

    warnings.warn("Reverting to legacy config file writing..", stacklevel=2)
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(cfg.get("multianimalproject", False))
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if "skeleton" not in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)


def edit_config(configname: str | Path, edits: dict, output_name: str | Path = "") -> dict:
    """
    Convenience function to edit and save a config file from a dictionary.

    Note:
        Legacy helper without schema validation. Prefer manipulating and saving
        the typed config instead (e.g. cfg.update(edits).to_yaml(cfg_path))

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
        warnings.warn("Some edits could not be written. The configuration file will be left unchanged.", stacklevel=2)
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


def write_config_3d_template(projconfigfile: str | Path, cfg_file_3d: dict, ruamelFile_3d: YAML) -> None:
    """Write 3D config from pre-built template and YAML instance."""
    with open(projconfigfile, "w") as cf:
        ruamelFile_3d.dump(cfg_file_3d, cf)
