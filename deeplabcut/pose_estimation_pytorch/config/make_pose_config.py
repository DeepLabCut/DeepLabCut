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
"""Methods to create the configuration files for PyTorch DeepLabCut models"""
from __future__ import annotations

import copy
from pathlib import Path

from deeplabcut.pose_estimation_pytorch.config.utils import (
    get_config_folder_path,
    load_backbones,
    load_base_config,
    read_config_as_dict,
    replace_default_values,
    update_config,
)
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal


def make_pytorch_pose_config(
    project_config: dict,
    pose_config_path: str,
    net_type: str | None = None,
    top_down: bool = False,
    weight_init: WeightInitialization | None = None,
) -> dict:
    """Creates a PyTorch pose configuration file for a DeepLabCut project

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
        project_config: the DeepLabCut project config
        pose_config_path: the path where the pytorch pose configuration will be saved
        net_type: the architecture of the desired pose estimation model
        top_down: when the net_type is a backbone, whether to create a top-down model
            by associating a detector to the pose model. Required for multi-animal
            projects when net_type is a backbone (as a backbone + heatmap head can only
            predict pose for single individuals).
        weight_init: Specify how model weights should be initialized. If None, ImageNet
            pretrained weights from Timm will be loaded when training.

    Returns:
        the PyTorch pose configuration file
    """
    multianimal_project = project_config.get("multianimalproject", False)
    individuals = project_config.get("individuals", ["single"])
    with_identity = project_config.get("identity")
    bodyparts = auxiliaryfunctions.get_bodyparts(project_config)
    unique_bpts = auxiliaryfunctions.get_unique_bodyparts(project_config)

    if net_type is None:
        net_type = project_config.get("default_net_type", "resnet_50")

    configs_dir = get_config_folder_path()
    pose_config = load_base_config(configs_dir)
    pose_config = add_metadata(project_config, pose_config, pose_config_path)
    pose_config["net_type"] = net_type

    backbones = load_backbones(configs_dir)
    if net_type in backbones:
        if not top_down and multianimal_project:
            model_cfg = create_backbone_with_paf_model(
                configs_dir=configs_dir,
                net_type=net_type,
                num_individuals=len(individuals),
                bodyparts=bodyparts,
                paf_parameters=_get_paf_parameters(project_config, bodyparts),
            )
        else:
            model_cfg = create_backbone_with_heatmap_model(
                configs_dir=configs_dir,
                net_type=net_type,
                multianimal_project=multianimal_project,
                bodyparts=bodyparts,
                top_down=top_down,
            )
    else:
        architecture = net_type.split("_")[0]
        default_value_kwargs = {}
        if architecture == "dlcrnet":
            default_value_kwargs.update(_get_paf_parameters(project_config, bodyparts))

        cfg_path = configs_dir / architecture / f"{net_type}.yaml"
        model_cfg = read_config_as_dict(cfg_path)
        model_cfg = replace_default_values(
            model_cfg,
            num_bodyparts=len(bodyparts),
            num_individuals=len(individuals),
            **default_value_kwargs,
        )

    is_top_down = model_cfg.get("method", "BU").upper() == "TD"
    if is_top_down:
        # FIXME(niels): Currently, the variant used is the default MobileNet. In the
        #  future, we want users to be able to choose which detector variant they use
        #  when creating the configuration file, instead of having to update it once
        #  created
        variant = None
        if weight_init is not None:
            # FIXME(niels): We only have fasterrcnn_resnet50_fpn_v2 SuperAnimal weights.
            #  This should be updated once more SuperAnimal detectors are uploaded,
            #  so that users can choose which pre-trained detector they use.
            variant = "fasterrcnn_resnet50_fpn_v2"

        model_cfg = add_detector(
            configs_dir,
            model_cfg,
            len(individuals),
            variant=variant,
        )

    # add the default augmentations to the config
    aug_filename = "aug_top_down.yaml" if is_top_down else "aug_default.yaml"
    aug_cfg = {"data": read_config_as_dict(configs_dir / "base" / aug_filename)}
    pose_config = update_config(pose_config, aug_cfg)

    # add the model to the config
    pose_config = update_config(pose_config, model_cfg)

    # set the dataset from which to load weights
    if weight_init is not None:
        pose_config["train_settings"]["weight_init"] = weight_init.to_dict()

    # add a unique bodypart head if needed
    if len(unique_bpts) > 0:
        if is_top_down:
            raise ValueError(
                f"You selected a top-down model architecture ({net_type}), but you have"
                f" unique bodyparts, which is not yet implemented for top-down models."
                " Please select a bottom-up architecture such as `resnet_50` for single"
                " animal projects or `dlcrnet_50` for multi-animal projects."
            )

        pose_config = add_unique_bodypart_head(
            configs_dir,
            pose_config,
            num_unique_bodyparts=len(unique_bpts),
            backbone_output_channels=pose_config["model"]["backbone_output_channels"],
        )

    # add an identity head if needed
    if with_identity:
        if is_top_down:
            raise ValueError(
                f"You selected a top-down model architecture ({net_type}), but you have"
                f" set `identity: true`, which is not yet implemented for top-down"
                f" models. Please select a bottom-up architecture such as `dlcrnet_50`"
                f" to train with identity, or set `identity: false`."
            )

        pose_config = add_identity_head(
            configs_dir,
            pose_config,
            num_individuals=len(individuals),
            backbone_output_channels=pose_config["model"]["backbone_output_channels"],
        )

    # sort first-level keys to make it prettier
    return dict(sorted(pose_config.items()))


def add_metadata(project_config: dict, config: dict, pose_config_path: str) -> dict:
    """Adds metadata to a pytorch pose configuration

    Args:
        project_config: the project configuration
        config: the pytorch pose configuration
        pose_config_path: the path where the pytorch pose configuration will be saved

    Returns:
        the configuration with a `meta` key added
    """
    config = copy.deepcopy(config)
    config["metadata"] = {
        "project_path": project_config["project_path"],
        "pose_config_path": pose_config_path,
        "bodyparts": auxiliaryfunctions.get_bodyparts(project_config),
        "unique_bodyparts": auxiliaryfunctions.get_unique_bodyparts(project_config),
        "individuals": project_config.get("individuals", ["animal"]),
        "with_identity": project_config.get("identity", False),
    }
    return config


def create_backbone_with_heatmap_model(
    configs_dir: Path,
    net_type: str,
    multianimal_project: bool,
    bodyparts: list[str],
    top_down: bool,
) -> dict:
    """
    Creates a simple heatmap pose estimation model, composed of a backbone and a head
    predicting heatmaps and location refinement maps

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
            f" {net_type} or use a detector to create a top-down model (create your"
            f" configuration with `make_pytorch_pose_config(..., top_down=True)`)."
        )

    # add the backbone to the config
    model_config = read_config_as_dict(configs_dir / "backbones" / f"{net_type}.yaml")
    backbone_output_channels = model_config["model"]["backbone_output_channels"]

    model_config["method"] = "bu"
    bodypart_head_name = "head_bodyparts.yaml"
    if top_down:
        model_config["method"] = "td"
        bodypart_head_name = "head_topdown.yaml"

    # add a bodypart head
    bodypart_head_config = read_config_as_dict(
        configs_dir / "base" / bodypart_head_name
    )
    model_config["model"]["heads"] = {
        "bodypart": replace_default_values(
            bodypart_head_config,
            num_bodyparts=len(bodyparts),
            backbone_output_channels=backbone_output_channels,
        )
    }
    return model_config


def create_backbone_with_paf_model(
    configs_dir: Path,
    net_type: str,
    num_individuals: int,
    bodyparts: list[str],
    paf_parameters: dict,
) -> dict:
    """
    Creates a pose estimation model, composed of a backbone and a head predicting
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
    model_config = read_config_as_dict(configs_dir / "backbones" / f"{net_type}.yaml")
    backbone_output_channels = model_config["model"]["backbone_output_channels"]

    # add a bodypart head
    bodypart_head_config = read_config_as_dict(
        configs_dir / "base" / f"head_bodyparts_with_paf.yaml"
    )
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


def add_detector(
    configs_dir: Path,
    config: dict,
    num_individuals: int,
    variant: str | None = None
) -> dict:
    """Adds a detector to a model

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        config: model configuration to update
        num_individuals: the maximum number of individuals the model should detect
        variant: the detector variant to use (if None, uses the variant set in the
            default detector.yaml config)

    Returns:
        the model configuration with an added detector config
    """
    config = copy.deepcopy(config)
    detector_config = read_config_as_dict(configs_dir / "base" / "detector.yaml")
    detector_config = replace_default_values(
        detector_config,
        num_individuals=num_individuals,
    )
    if variant is not None:
        detector_config["detector"]["model"]["variant"] = variant

    config = update_config(config, detector_config)
    return config


def add_unique_bodypart_head(
    configs_dir: Path,
    config: dict,
    num_unique_bodyparts: int,
    backbone_output_channels: int,
) -> dict:
    """Adds a unique bodypart head to a model

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


def add_identity_head(
    configs_dir: Path,
    config: dict,
    num_individuals: int,
    backbone_output_channels: int,
) -> dict:
    """Adds an identity head to a model

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


def _get_paf_parameters(
    project_config: dict,
    bodyparts: list[str],
    num_limbs_threshold: int = 105,
    paf_graph_degree: int = 6,
) -> dict:
    """Gets values for PAF parameters from the project configuration"""
    paf_graph = [
        [i, j] for i in range(len(bodyparts)) for j in range(i + 1, len(bodyparts))
    ]
    num_limbs = len(paf_graph)
    # If the graph is unnecessarily large (with 15+ keypoints by default),
    # we randomly prune it to a size guaranteeing an average node degree of 6;
    # see Suppl. Fig S9c in Lauer et al., 2022.
    if num_limbs >= num_limbs_threshold:
        paf_graph = auxfun_multianimal.prune_paf_graph(
            paf_graph,
            average_degree=paf_graph_degree,
        )
        num_limbs = len(paf_graph)
    return {
        "paf_graph": paf_graph,
        "num_limbs": num_limbs,
        "paf_edges_to_keep": project_config.get("paf_best", list(range(num_limbs))),
    }
