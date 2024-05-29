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
from __future__ import annotations

import os
import warnings
from glob import glob
from pathlib import Path

from deeplabcut.core.conversion_table import ConversionTable
from deeplabcut.utils.auxiliaryfunctions import (
    get_bodyparts,
    get_deeplabcut_path,
    read_config,
    write_config,
)


def dlc_modelzoo_path() -> Path:
    """Returns: the path to the `modelzoo` folder in the DeepLabCut installation"""
    dlc_root_path = Path(get_deeplabcut_path())
    return dlc_root_path / "modelzoo"


def get_super_animal_project_cfg(super_animal: str) -> dict:
    """Gets the project configuration file for a SuperAnimal model

    Args:
        super_animal: the name of the SuperAnimal model for which to load the project
            configuration

    Returns:
        the project configuration for the given SuperAnimal model

    Raises:
        ValueError if no such SuperAnimal is found
    """
    project_configs_dir = dlc_modelzoo_path() / "project_configs"
    super_animal_projects = {p.stem: p for p in project_configs_dir.iterdir()}
    if super_animal not in super_animal_projects:
        raise ValueError(
            f"No such SuperAnimal model: {super_animal}. Available SuperAnimal models "
            f"are {', '.join(super_animal_projects.keys())}."
        )

    return read_config(str(super_animal_projects[super_animal]))


def create_conversion_table(
    config: str | Path,
    super_animal: str,
    project_to_super_animal: dict[str, str],
) -> ConversionTable:
    """
    Creates a conversion table mapping bodyparts defined for a DeepLabCut project
    to bodyparts defined for a SuperAnimal model. This allows to fine-tune SuperAnimal
    weights instead of transfer learning from ImageNet. The conversion table is directly
    added to the project's configuration file.

    Args:
        config: The path to the project configuration for which the conversion table
            should be created.
        super_animal: The SuperAnimal model for the conversion table
        project_to_super_animal: The conversion table mapping each project bodypart
            to the corresponding SuperAnimal bodypart.

    Returns:
        The conversion table that was added to the project config.

    Raises:
         ValueError: If the conversion table is misconfigured (e.g., if there are
            misnamed bodyparts in the table). See ConversionTable for more.
    """
    cfg = read_config(str(config))
    sa_cfg = get_super_animal_project_cfg(super_animal)
    conversion_table = ConversionTable(
        super_animal=super_animal,
        project_bodyparts=get_bodyparts(cfg),
        super_animal_bodyparts=sa_cfg["bodyparts"],
        table=project_to_super_animal,
    )

    conversion_tables = cfg.get("SuperAnimalConversionTables")
    if conversion_tables is None:
        conversion_tables = {}

    conversion_tables[super_animal] = conversion_table.table
    cfg["SuperAnimalConversionTables"] = conversion_tables
    write_config(str(config), cfg)
    return conversion_table


def get_conversion_table(cfg: dict | str | Path, super_animal: str) -> ConversionTable:
    """Gets the conversion table from a project to a SuperAnimal model

    Args:
        cfg: The path to a project configuration file, or directly the project config.
        super_animal: The SuperAnimal for which to get the configuration file.

    Returns:
        A dictionary mapping {project_bodypart: super_animal_bodypart}

    Raises:
        ValueError: If the conversion table is misconfigured (e.g., if there are
            misnamed bodyparts in the table). See ConversionTable for more.
    """
    if isinstance(cfg, (str, Path)):
        cfg = read_config(str(cfg))

    conversion_tables = cfg.get("SuperAnimalConversionTables", {})
    if super_animal not in conversion_tables:
        raise ValueError(
            f"No conversion table defined in the project config for {super_animal}."
            "Call deeplabcut.modelzoo.create_conversion_table to create one."
        )

    sa_cfg = get_super_animal_project_cfg(super_animal)
    conversion_table = ConversionTable(
        super_animal=super_animal,
        project_bodyparts=get_bodyparts(cfg),
        super_animal_bodyparts=sa_cfg["bodyparts"],
        table=conversion_tables[super_animal],
    )
    return conversion_table


def parse_project_model_name(superanimal_name: str) -> tuple[str, str]:
    """Parses model zoo model names for SuperAnimal models

     Args:
         superanimal_name: the name of the SuperAnimal model name to parse

     Returns:
         project_name: the parsed SuperAnimal model name
         model_name: the model architecture (e.g., dlcrnet, hrnetw32)
     """

    if superanimal_name == "superanimal_quadruped":
        warnings.warn(
            f"{superanimal_name} is deprecated and will be removed in a future version. Use {superanimal_name}_model_suffix instead.",
            DeprecationWarning,
        )
        superanimal_name = "superanimal_quadruped_hrnetw32"

    if superanimal_name == "superanimal_topviewmouse":
        warnings.warn(
            f"{superanimal_name} is deprecated and will be removed in a future version. Use {superanimal_name}_model_suffix instead.",
            DeprecationWarning,
        )
        superanimal_name = "superanimal_topviewmouse_dlcrnet"

    model_name = superanimal_name.split("_")[-1]
    project_name = superanimal_name.replace(f"_{model_name}", "")

    dlc_root_path = get_deeplabcut_path()
    modelzoo_path = os.path.join(dlc_root_path, "modelzoo")

    available_model_configs = glob(
        os.path.join(modelzoo_path, "model_configs", "*.yaml")
    )
    available_models = [
        os.path.splitext(os.path.basename(path))[0] for path in available_model_configs
    ]

    if model_name not in available_models:
        raise ValueError(
            f"Model {model_name} not found. Available models are: {available_models}"
        )

    available_project_configs = glob(
        os.path.join(modelzoo_path, "project_configs", "*.yaml")
    )
    available_projects = [
        os.path.splitext(os.path.basename(path))[0]
        for path in available_project_configs
    ]

    return project_name, model_name
