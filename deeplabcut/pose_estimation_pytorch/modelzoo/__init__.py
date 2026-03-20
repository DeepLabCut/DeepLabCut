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
"""Public API for PyTorch modelzoo.

Exports are resolved lazily to avoid import cycles between helpers and package
initialization.
"""

from importlib import import_module

_EXPORTS = {
    "download_super_animal_snapshot": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "download_super_animal_snapshot",
    ),
    "get_snapshot_folder_path": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "get_snapshot_folder_path",
    ),
    "get_super_animal_model_config_path": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "get_super_animal_model_config_path",
    ),
    "get_super_animal_project_config_path": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "get_super_animal_project_config_path",
    ),
    "get_super_animal_snapshot_path": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "get_super_animal_snapshot_path",
    ),
    "load_super_animal_config": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.utils",
        "load_super_animal_config",
    ),
    "create_superanimal_inference_runners": (
        "deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers",
        "create_superanimal_inference_runners",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
