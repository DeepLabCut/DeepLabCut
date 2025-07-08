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

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from deeplabcut.core.config import read_config_as_dict
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

    return read_config_as_dict(super_animal_projects[super_animal])


def get_super_animal_scorer(
    super_animal: str,
    model_snapshot_path: Path,
    detector_snapshot_path: Path | str | None,
) -> str:
    """
    Args:
        super_animal: The SuperAnimal dataset on which the models were trained
        model_snapshot_path: The path for the SuperAnimal pose model snapshot
        detector_snapshot_path: The path or name for the SuperAnimal detector, if a detector is being used.

    Returns:
        The DLC scorer name to use for the given SuperAnimal models.
    """
    super_animal_prefix = super_animal + "_"
    # Always use model name first
    model_name = model_snapshot_path.stem if hasattr(model_snapshot_path, "stem") else str(model_snapshot_path)
    if model_name.startswith(super_animal_prefix):
        model_name = model_name[len(super_animal_prefix) :]
    dlc_scorer = f"{super_animal_prefix}{model_name}"

    # Then add detector name if provided
    if detector_snapshot_path:
        if isinstance(detector_snapshot_path, (str, Path)):
            detector_name = Path(detector_snapshot_path).stem if hasattr(detector_snapshot_path, "stem") else str(detector_snapshot_path)
            if detector_name.startswith(super_animal_prefix):
                detector_name = detector_name[len(super_animal_prefix) :]
            dlc_scorer += f"_{detector_name}"

    return dlc_scorer


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
    if conversion_tables is None or super_animal not in conversion_tables:
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


def read_conversion_table_from_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, header=None)
    df = df.dropna()
    df[0] = df[0].str.replace(r"\s+", "", regex=True)
    df[1] = df[1].str.replace(r"\s+", "", regex=True)
    _map = dict(zip(df[0], df[1]))
    return _map


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


def get_superanimal_colormaps():
    # FIXME(shaokai) - Add colormaps for the SuperBird dataset
    superanimal_bird_colors = (
        np.array(
            [
                (127, 0, 255),
                (115, 18, 254),
                (103, 37, 254),
                (91, 56, 253),
                (79, 74, 252),
                (65, 95, 250),
                (53, 112, 248),
                (41, 128, 246),
                (29, 144, 243),
                (15, 162, 239),
                (3, 176, 236),
                (8, 189, 232),
                (20, 201, 228),
                (34, 214, 223),
                (46, 223, 219),
                (58, 232, 214),
                (70, 239, 209),
                (84, 246, 202),
                (96, 250, 196),
                (108, 253, 190),
                (120, 254, 184),
                (134, 254, 176),
                (146, 253, 169),
                (158, 250, 162),
                (170, 246, 154),
                (184, 239, 146),
                (196, 232, 138),
                (208, 223, 130),
                (220, 214, 122),
                (234, 201, 112),
                (246, 189, 103),
                (255, 176, 95),
                (255, 162, 86),
                (255, 144, 75),
                (255, 128, 66),
                (255, 112, 57),
                (255, 95, 48),
                (255, 74, 37),
                (255, 56, 28),
                (255, 37, 18),
                (255, 18, 9),
                (255, 0, 0),
            ]
        )
        / 255
    )
    superanimal_topviewmouse_colors = (
        np.array(
            [
                [127, 0, 255],
                [109, 28, 254],
                [91, 56, 253],
                [71, 86, 251],
                [53, 112, 248],
                [33, 139, 244],
                [15, 162, 239],
                [4, 185, 234],
                [22, 203, 228],
                [42, 220, 220],
                [60, 233, 213],
                [80, 244, 204],
                [98, 250, 195],
                [118, 254, 185],
                [136, 254, 175],
                [156, 250, 163],
                [174, 244, 152],
                [194, 233, 139],
                [212, 220, 127],
                [232, 203, 113],
                [250, 185, 100],
                [255, 162, 86],
                [255, 139, 72],
                [255, 112, 57],
                [255, 86, 43],
                [255, 56, 28],
                [255, 28, 14],
            ]
        )
        / 255
    )
    superanimal_quadruped_colors = (
        np.array(
            [
                [255.0, 0.0, 0.0],
                [255.0, 39.63408568671726, 0.0],
                [255.0, 79.26817137343453, 0.0],
                [255.0, 118.9022570601518, 0.0],
                [255.0, 158.53634274686905, 0.0],
                [255.0, 198.17042843358632, 0.0],
                [255.0, 237.8045141203036, 0.0],
                [232.56140019297916, 255.0, 0.0],
                [192.92731450626187, 255.0, 0.0],
                [153.2932288195446, 255.0, 0.0],
                [113.65914313282731, 255.0, 0.0],
                [74.02505744611004, 255.0, 0.0],
                [34.390971759392784, 255.0, 0.0],
                [3.5647953575585385, 255.0, 8.807909284882923],
                [0.0, 255.0, 44.87701729490043],
                [0.0, 255.0, 84.51085328820125],
                [0.0, 255.0, 124.14468928150207],
                [0.0, 255.0, 163.77852527480275],
                [0.0, 255.0, 203.4123612681037],
                [0.0, 255.0, 243.04619726140453],
                [0, 220, 255],
                [0, 255, 255],
                [0, 165, 255],
                [0, 150, 255],
                [0.0, 68.78344961404169, 255.0],
                [0.0, 29.14936392732455, 255.0],
                [10.484721759392611, 0.0, 255.0],
                [50.11880744611004, 0.0, 255.0],
                [89.75289313282732, 0.0, 255.0],
                [129.38697881954448, 0.0, 255.0],
                [169.02106450626192, 0.0, 255.0],
                [169.02106450626192, 0.0, 255.0],
                [255.0, 0.0, 142.80850706015173],
                [169.02106450626192, 0.0, 255.0],
                [255.0, 0.0, 142.80850706015173],
                [255.0, 0.0, 142.80850706015173],
                [255.0, 0.0, 103.17442137343447],
                [255.0, 0.0, 63.54033568671722],
                [255.0, 0.0, 23.90625],
            ]
        )
        / 255
    )
    superanimal_humanbody_colors = (
        np.array(
            [
                [255, 0, 0],
                [255, 20, 0],
                [255, 40, 0],
                [255, 60, 0],
                [255, 80, 0],
                [255, 100, 0],
                [255, 120, 0],
                [255, 140, 0],
                [255, 160, 0],
                [255, 180, 0],
                [255, 200, 0],
                [255, 220, 0],
                [255, 240, 0],
                [255, 255, 0],
                [220, 255, 0],
                [180, 255, 0],
                [140, 255, 0],
                [100, 255, 0],
                [60, 255, 0],
                [20, 255, 0],
                [0, 255, 0],
                [0, 255, 40],
                [0, 255, 80],
                [0, 255, 120],
                [0, 255, 160],
                [0, 255, 200],
                [0, 255, 240],
                [0, 220, 255],
                [0, 180, 255],
                [0, 140, 255],
                [0, 100, 255],
                [0, 60, 255],
                [0, 20, 255],
                [0, 0, 255],
                [40, 0, 255],
                [80, 0, 255],
                [120, 0, 255],
                [160, 0, 255],
                [200, 0, 255],
                [240, 0, 255],
                [255, 0, 240],
                [255, 0, 200],
                [255, 0, 160],
                [255, 0, 120],
                [255, 0, 80],
                [255, 0, 40],
            ]
        )
        / 255
    )

    superanimal_colormaps = {
        "superanimal_bird": ListedColormap(
            list(superanimal_bird_colors), name="superanimal_bird"
        ),
        "superanimal_topviewmouse": ListedColormap(
            list(superanimal_topviewmouse_colors), name="superanimal_topviewmouse"
        ),
        "superanimal_quadruped": ListedColormap(
            list(superanimal_quadruped_colors), name="superanimal_quadruped"
        ),
        "superanimal_humanbody": ListedColormap(
            list(superanimal_humanbody_colors), name="superanimal_humanbody"
        ),
    }
    return superanimal_colormaps
