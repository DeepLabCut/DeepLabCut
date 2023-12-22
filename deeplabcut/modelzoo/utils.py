#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import json
import os
import warnings
from glob import glob

from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path


def parse_project_model_name(superanimal_name: str) -> str:
    """
    TODO

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
