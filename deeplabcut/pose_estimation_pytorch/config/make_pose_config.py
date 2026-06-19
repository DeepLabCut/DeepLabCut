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
"""Methods to create the configuration files for PyTorch DeepLabCut models."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deeplabcut.core.config import ProjectConfig, read_config_as_dict
from deeplabcut.core.deprecation import deprecated
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.config.enums import DetectorType, NetType
from deeplabcut.pose_estimation_pytorch.config.inference import InferenceConfig
from deeplabcut.pose_estimation_pytorch.config.metadata import PoseMetadata
from deeplabcut.pose_estimation_pytorch.config.paf_parameters import PAFParameters
from deeplabcut.pose_estimation_pytorch.config.utils import (
    get_config_folder_path,
    load_backbones,
    load_base_config,
    replace_default_values,
)
from deeplabcut.pose_estimation_pytorch.task import Task

if TYPE_CHECKING:
    from deeplabcut.pose_estimation_pytorch.config import DetectorConfig, PoseConfig, TestConfig

logger = logging.getLogger(__name__)


@deprecated(replacement="pose_estimation_pytorch.config.TestConfig.build", since="3.1")
def make_pytorch_test_config(
    model_config: PoseConfig | dict | str | Path,
    test_config_path: str | Path,
    save: bool = False,
) -> TestConfig:
    from deeplabcut.pose_estimation_pytorch.config import TestConfig

    return TestConfig.build(model_config, test_config_path=test_config_path, save=save)


@deprecated(replacement="pose_estimation_pytorch.config.PoseConfig.build", since="3.1")
def make_pytorch_pose_config(
    project_config: ProjectConfig | dict | Path | str,
    pose_config_path: str | Path,
    net_type: NetType | str | None = None,
    top_down: bool = False,
    detector_type: str | None = None,
    weight_init: WeightInitialization | None = None,
    save: bool = False,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
) -> PoseConfig:
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig

    return PoseConfig.build(
        project_config,
        pose_config_path,
        net_type=net_type,
        top_down=top_down,
        detector_type=detector_type,
        weight_init=weight_init,
        save=save,
        ctd_conditions=ctd_conditions,
    )


@deprecated(replacement="pose_estimation_pytorch.config.PoseMetadata", since="3.1")
def make_basic_project_config(
    dataset_path: Path | str,
    bodyparts: list[str],
    max_individuals: int,
    multi_animal: bool = True,
) -> dict:
    """Deprecated factory for basic config dict for non-DLC projects."""
    return PoseMetadata(
        project_path=dataset_path,
        bodyparts=bodyparts,
        individuals=[f"individual{i:03d}" for i in range(max_individuals)],
    ).to_dict_legacy()


def build_pose_config_defaults(
    net_type: NetType,
    metadata: PoseMetadata,
    *,
    task: Task,
    multi_animal: bool,
    paf_parameters: PAFParameters | None = None,
    weight_init: WeightInitialization | None = None,
    detector_config: DetectorConfig | None = None,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
) -> dict:
    """
    Load the model config defaults (from model-specific yaml) for a given project and net_type.

    The base/ folder contains default configurations, such as data augmentations or
    heatmap heads (that can be used to predict pose or identity based on visual
    features). These files are used to create pose model configurations.

    All available backbone configurations are stored in the backbones/ folder.
        - any backbone can be a single animal model with a heatmap head added on top
        - any backbone can be a top-down model with a detector and a heatmap head
        - any backbone can be a bottom-up model with a detector and a heatmap + PAF head

    All other model architectures have their own folders, with different variants
    available. Top-down model architectures must specify `method: TD` in their
    configuration files, from which this method adds a backbone configuration.

    Placeholder values (such as `num_bodyparts` or `num_individuals`) are filled in
    based on the project config file.

    Args:
        project_config: the DeepLabCut project config (used to infer individuals, bodyparts and identity tracking)
        net_type: the architecture of the desired pose estimation model
        task: when the net_type is a backbone, whether to create a top-down model
            by associating a detector to the pose model. Required for multi-animal
            projects when net_type is a backbone (as a backbone + heatmap head can only
            predict pose for single individuals).
        detector_type: for top-down pose models, the architecture of the desired object
            detection model
        ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] , optional, default = None,
            If using a conditional-top-down (CTD) net_type, this argument needs to be specified.
            It defines the conditions that will be used with the CTD model.
            It can be either:
                * A shuffle number (ctd_conditions: int), which must correspond to a bottom-up (BU) network type.
                * A predictions file path (ctd_conditions: string | Path), which must correspond to a .json or .h5
                predictions file.
                * A shuffle number and a particular snapshot (ctd_conditions: tuple[int, str] | tuple[int, int]), which
                respectively correspond to a bottom-up (BU) network type and a particular snapshot name or index.


    Returns:
        The model configuration defaults as a dictionary.
    """
    configs_dir = get_config_folder_path()
    base_cfg = load_base_config(configs_dir)
    backbones = load_backbones(configs_dir)

    if net_type in backbones:
        if task == Task.BOTTOM_UP and multi_animal:
            if paf_parameters is None:
                raise ValueError("PAF parameters are required for multi-animal bottom-up models.")
            model_cfg = _create_backbone_with_paf_model(
                configs_dir=configs_dir,
                net_type=net_type,
                num_individuals=metadata.num_individuals,
                bodyparts=metadata.bodyparts,
                paf_parameters=paf_parameters.to_dict(),
            )
        else:
            model_cfg = _create_backbone_with_heatmap_model(
                configs_dir=configs_dir,
                net_type=net_type,
                multianimal_project=multi_animal,
                bodyparts=metadata.bodyparts,
                top_down=task == Task.TOP_DOWN,
            )
    else:
        architecture = net_type.value.split("_")[0]
        default_value_kwargs = {}
        if architecture == "dlcrnet":
            if paf_parameters is None:
                raise ValueError("PAF parameters are required for DLCRNet models.")
            default_value_kwargs.update(paf_parameters.to_dict())

        cfg_path = configs_dir / architecture / f"{net_type.value}.yaml"
        model_cfg = read_config_as_dict(cfg_path)
        model_cfg = replace_default_values(
            model_cfg,
            num_bodyparts=metadata.num_bodyparts,
            num_individuals=metadata.num_individuals,
            **default_value_kwargs,
        )
    model_cfg["net_type"] = net_type.value

    if task == Task.TOP_DOWN:
        if detector_config is None:
            raise ValueError("detector_config is required for top-down pose configs.")
        model_cfg["detector"] = detector_config.to_dict()

    # add the default augmentations to the config
    aug_filename = "aug_default.yaml" if task == Task.BOTTOM_UP else "aug_top_down.yaml"
    aug_cfg = {"data": read_config_as_dict(configs_dir / "base" / aug_filename)}

    model_cfg = _update_config(model_cfg, aug_cfg)
    model_cfg = _update_config(base_cfg, model_cfg)

    # add a unique bodypart head if needed
    if metadata.unique_bodyparts:
        if task != Task.BOTTOM_UP:
            raise ValueError(
                f"You selected a top-down model architecture ({net_type.value}), but you have"
                f" unique bodyparts, which is not yet implemented for top-down models."
                " Please select a bottom-up architecture such as `resnet_50` for single"
                " animal projects or `dlcrnet_50` for multi-animal projects."
            )

        model_cfg = _add_unique_bodypart_head(
            configs_dir,
            model_cfg,
            num_unique_bodyparts=metadata.num_unique_bodyparts,
            backbone_output_channels=model_cfg["model"]["backbone_output_channels"],
        )

    # add an identity head if needed
    if metadata.with_identity:
        if task != Task.BOTTOM_UP:
            raise ValueError(
                f"You selected a top-down model architecture ({net_type.value}), but you have"
                f" set `identity: true`, which is not yet implemented for top-down"
                f" models. Please select a bottom-up architecture such as `dlcrnet_50`"
                f" to train with identity, or set `identity: false`."
            )

        model_cfg = _add_identity_head(
            configs_dir,
            model_cfg,
            num_individuals=metadata.num_individuals,
            backbone_output_channels=model_cfg["model"]["backbone_output_channels"],
        )

    model_cfg["inference"] = InferenceConfig().to_dict()
    # Add conditions for CTD models if specified
    if task == Task.COND_TOP_DOWN:
        if ctd_conditions is None:
            raise ValueError("A CTD conditions is required for conditional-top-down models.")
        _add_ctd_conditions(model_cfg, ctd_conditions)

    # Add metadata and weight init to the model config
    model_cfg["metadata"] = metadata.to_dict()
    if weight_init is not None:
        model_cfg["train_settings"]["weight_init"] = weight_init.to_dict()
    return model_cfg


def build_detector_config_defaults(
    num_individuals: int,
    detector_type: DetectorType,
) -> dict:
    """Adds a detector to a model.

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        num_individuals: the maximum number of individuals the model should detect
        detector_type: the type of detector to use (if None, uses ``ssdlite``)

    Returns:
        the model configuration with an added detector config
    """
    configs_dir = get_config_folder_path()
    detector_config = _update_config(
        read_config_as_dict(configs_dir / "base" / "base_detector.yaml"),
        read_config_as_dict(configs_dir / "detectors" / f"{detector_type.value}.yaml"),
    )
    detector_config = replace_default_values(
        detector_config,
        num_individuals=num_individuals,
    )
    return dict(sorted(detector_config.items()))


# TODO @deruyter92 2026-06-12: currently, the responsibility for determining the task
# is controlled either in the default config (for "non-backbone" models) or by the
# API parameter `top_down` (for "backbone" models). This should be refactored.
def _resolve_task(net_type: NetType, *, top_down: bool) -> Task:
    configs_dir = get_config_folder_path()
    backbones = load_backbones(configs_dir)
    if net_type in backbones:
        return Task.TOP_DOWN if top_down else Task.BOTTOM_UP
    architecture = net_type.value.split("_")[0]
    cfg_path = configs_dir / architecture / f"{net_type.value}.yaml"
    model_cfg = read_config_as_dict(cfg_path)
    return Task(model_cfg.get("method", "BU").upper())


def resolve_net_type_and_task(
    net_type: str | NetType | None,
    *,
    default: str,
    top_down: bool,
) -> tuple[NetType, Task]:
    """Resolve the net type from build args and project config default.

    Args:
        net_type (str | None): Architecture name, or None to use ``default``
            from project config.
        default (str): Fallback when ``net_type`` is None. Invalid values warn
            and fall back to ``resnet_50``.
        top_down (bool): Build a top-down backbone (ignored for non-backbones).

    Returns:
        (NetType, Task): the resolved canonical NetType and Task
    """
    if net_type is None:
        try:
            net_type, td_prefix = NetType.from_alias(default)
        except ValueError as e:
            raise ValueError(
                f"Invalid default_net_type in project config: {default}. Must be one of {NetType.available_aliases()}"
            ) from e
    else:
        net_type, td_prefix = NetType.from_alias(str(net_type))  # fails loudly if invalid

    if td_prefix:
        if top_down:
            logger.warning(
                "Passed net_type with top_down prefix. Instead use "
                "PoseConfig.build(..., top_down=True) to specify the task."
            )
        else:
            raise ValueError(
                "Passed net_type with top_down prefix. but top_down is False."
                "Please use only PoseConfig.build(..., top_down=True/False) to specify the task."
            )
    task = _resolve_task(net_type, top_down=top_down)
    return net_type, task


# NOTE @deruyter92 2026-06-19: Moved this from public API to internal use only.
# Only used for construction of default config dictionaries. Can likely be refactored.
def _update_config(config: dict, updates: dict, copy_original: bool = True) -> dict:
    """Updates items in the configuration file.

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
                config[k] = _update_config(config[k], v, copy_original=False)
        else:
            config[k] = copy.deepcopy(v)
    return config


def _add_ctd_conditions(model_cfg: dict, ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int]):
    """
    Args:
        model_cfg: dict, contents of pytorch_config.yaml
        ctd_conditions: Only for using conditional-top-down (CTD) models. It defines
            the conditions that will be used with the CTD model. It can be:
            * A shuffle number (ctd_conditions: int), which must correspond to a
                bottom-up (BU) network type.
            * A predictions file path (ctd_conditions: string | Path), which must
                correspond to a .json or .h5 predictions file.
            * A shuffle number and a particular snapshot (ctd_conditions:
                tuple[int, str] | tuple[int, int]), which respectively correspond to a
                bottom-up (BU) network type and a particular snapshot name or index.
    """
    if isinstance(ctd_conditions, int):
        conditions = {"shuffle": ctd_conditions}

    elif isinstance(ctd_conditions, str) or isinstance(ctd_conditions, Path):
        ctd_conditions = Path(ctd_conditions)
        if not ctd_conditions.exists():
            raise FileNotFoundError(f"Invalid path: {ctd_conditions}")
        if ctd_conditions.suffix not in (".h5", ".json"):
            raise ValueError("Invalid conditions file extension.")
        conditions = str(ctd_conditions.resolve())

    elif isinstance(ctd_conditions, tuple):
        if len(ctd_conditions) != 2:
            raise ValueError("Invalid conditions tuple length.")
        if not isinstance(ctd_conditions[0], int):
            raise TypeError("Conditions shuffle number must be of type int.")
        if isinstance(ctd_conditions[1], int):
            conditions = {
                "shuffle": ctd_conditions[0],
                "snapshot_index": ctd_conditions[1],
            }
        elif isinstance(ctd_conditions[1], str):
            conditions = {"shuffle": ctd_conditions[0], "snapshot": ctd_conditions[1]}
        else:
            raise TypeError("Conditions snapshot must be of type int (index) or string (snapshot name).")
    else:
        raise TypeError("Conditions ctd_conditions is of invalid type.")

    model_cfg["inference"]["conditions"] = conditions


def _create_backbone_with_heatmap_model(
    configs_dir: Path,
    net_type: NetType,
    multianimal_project: bool,
    bodyparts: list[str],
    top_down: bool,
) -> dict:
    """Creates a simple heatmap pose estimation model, composed of a backbone and a head
    predicting heatmaps and location refinement maps.

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        net_type: the type of backbone to create the model with (e.g., resnet_50)
        multianimal_project: whether this model is created for a multi-animal project
        bodyparts: the bodyparts to detect
        top_down: whether the model will be associated to a detector to form a top-down
            pose estimation model

    Returns:
        the backbone + heatmap model configuration

    Raises:
        ValueError: if the model is being created for a multi-animal project but the
            head won't be associated with a detector (heatmaps can only predict
            bodyparts for a single individual).
    """
    if multianimal_project and not top_down:
        raise ValueError(
            "A pose model formed of a backbone and simple heatmap + location refinement"
            " head can only be used for single animal projects. As you're working with"
            " a multi-animal project, please select a multi-individual model instead of"
            f" {net_type.value} or use a detector to create a top-down model."
        )

    # add the backbone to the config
    model_config = read_config_as_dict(configs_dir / "backbones" / f"{net_type.value}.yaml")
    backbone_output_channels = model_config["model"]["backbone_output_channels"]

    model_config["method"] = "bu"
    bodypart_head_name = "head_bodyparts.yaml"
    if top_down:
        model_config["method"] = "td"
        bodypart_head_name = "head_topdown.yaml"

    # add a bodypart head
    bodypart_head_config = read_config_as_dict(configs_dir / "base" / bodypart_head_name)
    model_config["model"]["heads"] = {
        "bodypart": replace_default_values(
            bodypart_head_config,
            num_bodyparts=len(bodyparts),
            backbone_output_channels=backbone_output_channels,
        )
    }
    return model_config


def _create_backbone_with_paf_model(
    configs_dir: Path,
    net_type: NetType,
    num_individuals: int,
    bodyparts: list[str],
    paf_parameters: dict,
) -> dict:
    """Creates a pose estimation model, composed of a backbone and a head predicting
    heatmaps, location refinement maps and part affinity fields for multi-animal pose
    estimation.

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        net_type: the type of backbone to create the model with (e.g., resnet_50)
        num_individuals: the maximum number of individuals in a frame
        bodyparts: the bodyparts to detect
        paf_parameters: the parameters for the PAF

    Returns:
        the backbone + heatmap, location refinement, PAF model configuration
    """
    # add the backbone to the config
    model_config = read_config_as_dict(configs_dir / "backbones" / f"{net_type.value}.yaml")
    backbone_output_channels = model_config["model"]["backbone_output_channels"]

    # add a bodypart head
    bodypart_head_config = read_config_as_dict(configs_dir / "base" / "head_bodyparts_with_paf.yaml")
    model_config["model"]["heads"] = {
        "bodypart": replace_default_values(
            bodypart_head_config,
            num_bodyparts=len(bodyparts),
            num_individuals=num_individuals,
            backbone_output_channels=backbone_output_channels,
            **paf_parameters,
        )
    }
    return model_config


def _add_unique_bodypart_head(
    configs_dir: Path,
    config: dict,
    num_unique_bodyparts: int,
    backbone_output_channels: int,
) -> dict:
    """Adds a unique bodypart head to a model.

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        config: model configuration to update
        num_unique_bodyparts: the number of unique bodyparts to detect
        backbone_output_channels: the number of channels output by the model backbone

    Returns:
        the configuration with an added unique bodypart head
    """
    config = copy.deepcopy(config)
    unique_head_config = replace_default_values(
        read_config_as_dict(configs_dir / "base" / "head_bodyparts.yaml"),
        num_bodyparts=num_unique_bodyparts,
        backbone_output_channels=backbone_output_channels,
    )
    unique_head_config["target_generator"]["label_keypoint_key"] = "keypoints_unique"
    config["model"]["heads"]["unique_bodypart"] = unique_head_config
    return config


def _add_identity_head(
    configs_dir: Path,
    config: dict,
    num_individuals: int,
    backbone_output_channels: int,
) -> dict:
    """Adds an identity head to a model.

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        config: model configuration to update
        num_individuals: the number of individuals to re-identify
        backbone_output_channels: the number of channels output by the model backbone

    Returns:
        the configuration with an added identity head
    """
    config = copy.deepcopy(config)
    id_head_config = read_config_as_dict(configs_dir / "base" / "head_identity.yaml")
    config["model"]["heads"]["identity"] = replace_default_values(
        id_head_config,
        num_individuals=num_individuals,
        backbone_output_channels=backbone_output_channels,
    )
    return config
