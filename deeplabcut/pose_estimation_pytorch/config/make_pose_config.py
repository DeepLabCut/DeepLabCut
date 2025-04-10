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

from deeplabcut.core.config import read_config_as_dict, write_config
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.config.utils import (
    get_config_folder_path,
    load_backbones,
    load_base_config,
    replace_default_values,
    update_config,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal


def make_pytorch_pose_config(
    project_config: dict,
    pose_config_path: str | Path,
    net_type: str | None = None,
    top_down: bool = False,
    detector_type: str | None = None,
    weight_init: WeightInitialization | None = None,
    save: bool = False,
    ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
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
        detector_type: for top-down pose models, the architecture of the desired object
            detection model
        weight_init: Specify how model weights should be initialized. If None, ImageNet
            pretrained weights from Timm will be loaded when training.
        save: Whether to save the model configuration file to the ``pose_config_path``.
        ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] , optional, default = None,
            If using a conditional-top-down (CTD) net_type, this argument needs to be specified.
            It defines the conditions that will be used with the CTD model.
            It can be either:
                * A shuffle number (ctd_conditions: int), which must correspond to a bottom-up (BU) network type.
                * A predictions file path (ctd_conditions: string | Path), which must correspond to a .json or .h5 predictions file.
                * A shuffle number and a particular snapshot (ctd_conditions: tuple[int, str] | tuple[int, int]), which respectively correspond to a bottom-up (BU) network type and a particular snapshot name or index.


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

    task = Task(model_cfg.get("method", "BU").upper())
    if task == Task.TOP_DOWN:
        model_cfg = add_detector(
            configs_dir,
            model_cfg,
            len(individuals),
            detector_type=detector_type,
        )

    # add the default augmentations to the config
    aug_filename = "aug_default.yaml" if task == Task.BOTTOM_UP else "aug_top_down.yaml"
    aug_cfg = {"data": read_config_as_dict(configs_dir / "base" / aug_filename)}

    # Add conditions for CTD models if specified
    if task == Task.COND_TOP_DOWN and ctd_conditions is not None:
        _add_ctd_conditions(aug_cfg, ctd_conditions)

    pose_config = update_config(pose_config, aug_cfg)

    # add the model to the config
    pose_config = update_config(pose_config, model_cfg)

    # set the dataset from which to load weights
    if weight_init is not None:
        pose_config["train_settings"]["weight_init"] = weight_init.to_dict()

    # add a unique bodypart head if needed
    if len(unique_bpts) > 0:
        if task != Task.BOTTOM_UP:
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
        if task != Task.BOTTOM_UP:
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
    pose_config = dict(sorted(pose_config.items()))

    if save:
        write_config(pose_config_path, pose_config, overwrite=True)

    return pose_config


def _add_ctd_conditions(
    aug_cfg: dict, ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int]
):
    """
    Args:
        aug_cfg: dict, data augmentation configuration
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
            raise ValueError(f"Invalid conditions file extension.")
        conditions = str(ctd_conditions.resolve())

    elif isinstance(ctd_conditions, tuple):
        if len(ctd_conditions) != 2:
            raise ValueError(f"Invalid conditions tuple length.")
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
            raise TypeError(
                "Conditions snapshot must be of type int (index) or string (snapshot name)."
            )
    else:
        raise TypeError("Conditions ctd_conditions is of invalid type.")

    aug_cfg["data"]["conditions"] = conditions


def make_pytorch_test_config(
    model_config: dict,
    test_config_path: str | Path,
    save: bool = False,
) -> dict:
    """Creates the test configuration for a model

    Args:
        model_config: The PyTorch config for the model.
        test_config_path: The path of the test config
        save: Whether to save the test config to ``test_config_path``.

    Returns:
        The test configuration file.
    """
    bodyparts = model_config["metadata"]["bodyparts"]
    unique_bodyparts = model_config["metadata"]["unique_bodyparts"]
    all_joint_names = bodyparts + unique_bodyparts

    test_config = dict(
        dataset=model_config["metadata"]["project_path"],
        dataset_type="multi-animal-imgaug",  # required for downstream tracking
        num_joints=len(all_joint_names),
        all_joints=[[i] for i in range(len(all_joint_names))],
        all_joints_names=all_joint_names,
        net_type=model_config["net_type"],
        global_scale=1,
        scoremap_dir="test",
    )
    if save:
        write_config(test_config_path, test_config)

    return test_config


def make_basic_project_config(
    dataset_path: Path | str,
    bodyparts: list[str],
    max_individuals: int,
    multi_animal: bool = True,
) -> dict:
    """Creates a basic configuration dict that can be used to create model configs.

    This should be used to create the `project_config` given to
    `make_pytorch_pose_config` for non-DeepLabCut projects (e.g. when creating a
    configuration file for a model that will be trained on a COCO dataset).

    Args:
        dataset_path: The path to the dataset for which the config will be created.
        bodyparts: The bodyparts labeled for individuals in the dataset.
        max_individuals: The maximum number of individuals to detect in a single image.
        multi_animal: Whether multiple animals can be present in an image.

    Returns:
        The created project configuration dict that can be given to
        `make_pytorch_pose_config`.

    Examples:
        Creating a `pytorch_config` for a ResNet50 backbone with a part-affinity head (
        as multi_animal=True and top_down=False)

        >>> import deeplabcut.pose_estimation_pytorch as pep
        >>> project_config = pep.config.make_basic_project_config(
        >>>     dataset_path="/path/coco",
        >>>     bodyparts=["nose", "left_eye", "right_eye"],
        >>>     max_individuals=12,
        >>>     multi_animal=True,
        >>> )
        >>> model_config = pep.config.make_pytorch_pose_config(
        >>>     project_config=project_config,
        >>>     pose_config_path="/path/coco/models/resnet50/pytorch_config.yaml",
        >>>     net_type="resnet_50",
        >>>     top_down=False,
        >>>     save=True,
        >>> )

        Creating a `pytorch_config` for a ResNet50 backbone with a simple heatmap head
        (as the project is single-animal):

        >>> import deeplabcut.pose_estimation_pytorch as pep
        >>> project_config = pep.config.make_basic_project_config(
        >>>     dataset_path="/path/coco",
        >>>     bodyparts=["nose", "left_eye", "right_eye"],
        >>>     max_individuals=1,
        >>>     multi_animal=False,
        >>> )
        >>> model_config = pep.config.make_pytorch_pose_config(
        >>>     project_config=project_config,
        >>>     pose_config_path="/path/coco/models/resnet50/pytorch_config.yaml",
        >>>     net_type="resnet_50",
        >>>     top_down=False,
        >>>     save=True,
        >>> )
    """
    return dict(
        project_path=str(dataset_path),
        multianimalproject=multi_animal,
        bodyparts=bodyparts,
        multianimalbodyparts=bodyparts,
        uniquebodyparts=[],
        individuals=[f"individual{i:03d}" for i in range(max_individuals)],
    )


def add_metadata(
    project_config: dict, config: dict, pose_config_path: str | Path
) -> dict:
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
        "pose_config_path": str(pose_config_path),
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
    detector_type: str | None = None,
) -> dict:
    """Adds a detector to a model

    Args:
        configs_dir: path to the DeepLabCut "configs" directory
        config: model configuration to update
        num_individuals: the maximum number of individuals the model should detect
        detector_type: the type of detector to use (if None, uses ``ssdlite``)

    Returns:
        the model configuration with an added detector config
    """
    if detector_type is None:
        detector_type = "ssdlite"  # default detector

    detector_type = detector_type.lower()
    config = copy.deepcopy(config)
    detector_config = update_config(
        read_config_as_dict(configs_dir / "base" / "base_detector.yaml"),
        read_config_as_dict(configs_dir / "detectors" / f"{detector_type}.yaml"),
    )
    detector_config = replace_default_values(
        detector_config,
        num_individuals=num_individuals,
    )
    config["detector"] = dict(sorted(detector_config.items()))
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
