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
"""Main pose configuration class for DeepLabCut pose estimation models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deeplabcut.core.weight_init import WeightInitialization

from pydantic import Field, field_validator
from typing_extensions import Self

from deeplabcut.core.config import DLCBaseConfig, DLCVersionedConfig, ProjectConfig
from deeplabcut.core.config.validation import Fraction, NonNegativeInt, UniqueStrList
from deeplabcut.pose_estimation_pytorch.config.data import DataConfig, DetectorDataConfig
from deeplabcut.pose_estimation_pytorch.config.enums import DatasetType, DetectorType, MethodType, NetType
from deeplabcut.pose_estimation_pytorch.config.inference import InferenceConfig
from deeplabcut.pose_estimation_pytorch.config.logger import (
    CSVLoggerConfig,
    WandbLoggerConfig,
)
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import (
    build_detector_config_defaults,
    build_pose_config_defaults,
    resolve_net_type_and_task,
)
from deeplabcut.pose_estimation_pytorch.config.metadata import PoseMetadata
from deeplabcut.pose_estimation_pytorch.config.model import (
    DetectorModelConfig,
    ModelConfig,
)
from deeplabcut.pose_estimation_pytorch.config.paf_parameters import PAFParameters
from deeplabcut.pose_estimation_pytorch.config.runner import RunnerConfig
from deeplabcut.pose_estimation_pytorch.config.training import TrainSettingsConfig
from deeplabcut.pose_estimation_pytorch.task import Task


class DetectorConfig(DLCBaseConfig):
    model: DetectorModelConfig
    device: str = "auto"
    data: DetectorDataConfig | None = None
    runner: RunnerConfig | None = None
    train_settings: TrainSettingsConfig | None = None
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    resume_training_from: str | None = None

    @classmethod
    def build(
        cls,
        num_individuals: int,
        detector_type: DetectorType | None = None,
    ) -> Self:
        if detector_type is None:
            detector_type = DetectorType.SSDLITE

        return cls.from_dict(
            build_detector_config_defaults(
                num_individuals=num_individuals,
                detector_type=detector_type,
            )
        )


class PoseConfig(DLCVersionedConfig):
    """Main configuration class for DeepLabCut pose estimation models.

    This is the top-level configuration that brings together all the different
    configuration domains (project, model, data, training, etc.).

    Attributes:
        net_type: Network architecture type (e.g., resnet_50, hrnet_w32, dlcrnet_stride16_ms5)
        method: Method type (bu=Bottom-Up, td=Top-Down, ctd=Conditional Top-Down)
        device: Device configuration (auto, cpu, cuda)
        project: Project configuration (skeleton, individuals, etc.)
        model: Model configuration (backbone, heads, etc.)
        detector: Detector configuration (for top-down models)
        data: Data configuration (loaders, transforms, etc.)
        training: Training configuration (runner, optimizer, etc.)
        inference: Inference configuration (multithreading, compilation, etc.)
        logger: Logger configuration (e.g., WandB or CSV logger)
        with_center_keypoints: Whether to include center keypoints (for DEKR models)
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    net_type: NetType = NetType.RESNET_50
    method: MethodType = MethodType.BOTTOM_UP
    device: str = "auto"
    metadata: PoseMetadata
    data: DataConfig
    runner: RunnerConfig
    train_settings: TrainSettingsConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logger: CSVLoggerConfig | WandbLoggerConfig | None = Field(default=None, discriminator="type")
    with_center_keypoints: bool = False
    detector: DetectorConfig | None = None
    resume_training_from: str | None = None

    @field_validator("net_type", mode="before")
    @classmethod
    def _coerce_net_type(cls, v: object) -> NetType:
        if isinstance(v, NetType):
            return v
        net_type, _ = NetType.from_alias(str(v))
        return net_type

    @classmethod
    def build(
        cls,
        project_config: ProjectConfig | dict | Path | str,
        pose_config_path: str | Path,
        *,
        top_down: bool,
        multi_animal: bool | None = None,
        net_type: NetType | str | None = None,
        detector_type: DetectorType | str | None = None,
        weight_init: WeightInitialization | dict | Path | str | None = None,
        ctd_conditions: int | str | Path | tuple[int, str] | tuple[int, int] | None = None,
        save: bool = False,
    ) -> Self:
        """Build a typed PoseConfig for a project

        Args:
            project_config (ProjectConfig | dict | Path | str): The project configuration.
            pose_config_path (str | Path): The path to the pose configuration.
            top_down (bool): Whether to use a top-down backbone.
            net_type (NetType | str | None, optional): The network architecture type (without 'top_down_' prefix).
                If None, the default net type from the project config will be used.
            detector_type (DetectorType | str | None, optional): The detector architecture. Required for td models.
            weight_init (WeightInitialization | None, optional): The weight initialization object or path.
            ctd_conditions (int | str | Path | tuple[int, str] | tuple[int, int] | None, optional):
                The conditional top-down conditions. Only required for CTD models.
            save (bool, optional): Whether to save the pose configuration.

        Note:
            For generic backbone models, ``method`` is resolved from ``top_down``. For non-backbone models,
            the ``top_down`` is ignored and ``method`` is resolved from the default config.
        """
        from deeplabcut.core.weight_init import WeightInitialization

        # Normalize input parameters + set defaults if not provided
        project_config = ProjectConfig.from_any(project_config)
        if multi_animal is None:
            multi_animal = project_config.multianimalproject
        net_type, task = resolve_net_type_and_task(
            net_type,
            default=project_config.default_net_type,
            top_down=top_down,
        )
        if detector_type is not None:
            detector_type = DetectorType(detector_type)
        if weight_init is not None:
            weight_init = WeightInitialization.from_any(weight_init)

        # Build related configurations (PoseMetadata, PAFParameters, DetectorConfig)
        metadata = PoseMetadata.build(project_config, pose_config_path=pose_config_path)
        paf_parameters = None
        detector_config = None
        if task == Task.BOTTOM_UP and multi_animal:
            paf_parameters = PAFParameters.build(project_config)
        elif task == Task.TOP_DOWN:
            detector_config = DetectorConfig.build(metadata.num_individuals, detector_type)

        # Build the pose model config
        defaults: dict = build_pose_config_defaults(
            net_type=net_type,
            metadata=metadata,
            paf_parameters=paf_parameters,
            weight_init=weight_init,
            task=task,
            multi_animal=multi_animal,
            detector_config=detector_config,
            ctd_conditions=ctd_conditions,
        )
        pose_config = cls.from_dict(defaults)

        # Save if needed
        if save:
            pose_config.to_yaml(pose_config_path, overwrite=True)

        return pose_config

    @classmethod
    def build_for_superanimal_inference(
        cls,
        super_animal: str,
        *,
        model_name: str,
        detector_name: str | None = None,
        max_individuals: int = 30,
        device: str | None = None,
    ) -> Self:
        from deeplabcut.pose_estimation_pytorch.modelzoo.config import build_superanimal_inference_config

        metadata = PoseMetadata.build_for_superanimal(
            super_animal=super_animal, model_name=model_name, max_individuals=max_individuals
        )
        return cls.from_dict(
            build_superanimal_inference_config(
                super_animal=super_animal,
                model_name=model_name,
                detector_name=detector_name,
                metadata=metadata,
                device=device,
            )
        )

    @classmethod
    def build_for_superanimal_finetune(
        cls,
        project_config: ProjectConfig | dict | Path | str,
        *,
        model_name: str,
        detector_name: str | None,
        pose_config_path: str | Path,
        weight_init: WeightInitialization,
        inference_config: InferenceConfig | dict | Path | str | None = None,
        save: bool = False,
    ) -> Self:
        from deeplabcut.pose_estimation_pytorch.modelzoo.config import build_superanimal_finetune_config

        # Normalize input parameters + build related configurations
        project_config = ProjectConfig.from_any(project_config)
        if inference_config is None:
            inference_config = InferenceConfig()
        else:
            inference_config = InferenceConfig.from_any(inference_config)
        metadata = PoseMetadata.build(project_config, pose_config_path=pose_config_path)

        # Input validation
        if weight_init.dataset is None:
            raise ValueError("`WeightInitialization.dataset` is required for fine-tuning SuperAnimal models.")

        if not weight_init.with_decoder:
            raise ValueError(
                "`weight_init.with_decoder=True` is required for fine-tuning SuperAnimal models."
                "Please set `with_decoder=True` to fine-tune a model, or create a transfer learning config instead."
            )

        # Build the pose configuration
        pose_config = cls.from_dict(
            build_superanimal_finetune_config(
                weight_init,
                metadata,
                model_name,
                detector_name,
                inference_config=inference_config,
            )
        )

        if save:
            pose_config.to_yaml(metadata.pose_config_path, overwrite=True)
        return pose_config


class TestConfig(DLCBaseConfig):
    """Configuration class for DeepLabCut test/inference settings.

    This configuration is used for downstream tracking and evaluation, containing
    the essential metadata about joints and network architecture.

    Attributes:
        dataset: Path to the project/dataset.
        dataset_type: Type of dataset (required for downstream tracking).
        num_joints: Total number of joints (bodyparts + unique bodyparts).
        all_joints: List of joint indices, each as a single-element list.
        all_joints_names: List of joint names.
        net_type: Network architecture type.
        global_scale: Global scale factor for inference.
        scoremap_dir: Directory for score maps.
    """

    # TODO @deruyter92 2026-02-05: Is this additional configuration really needed?
    # We could aim for using the PoseConfig class or InferenceConfig class instead.
    dataset: Path = Path()
    num_joints: NonNegativeInt = 0
    all_joints: list[list[NonNegativeInt]] = Field(default_factory=list)
    all_joints_names: UniqueStrList = Field(default_factory=list)
    net_type: NetType = NetType.RESNET_50
    dataset_type: DatasetType = DatasetType.MULTIANIMAL_IMGAUG
    global_scale: Fraction = 1.0
    scoremap_dir: Path = Path()

    @classmethod
    def build(
        cls,
        pose_config: PoseConfig | dict | Path | str,
        *,
        dataset_type: DatasetType = DatasetType.MULTIANIMAL_IMGAUG,
        scoremap_dir: Path | str = "test",
        test_config_path: Path | str | None = None,
        global_scale: Fraction = 1.0,
        save: bool = False,
    ) -> Self:

        # Needs a validated PoseConfig
        cfg = PoseConfig.from_any(pose_config)
        metadata = cfg.metadata

        # Build the test config
        test_config = cls(
            dataset=cfg.metadata.project_path,
            dataset_type=dataset_type,  # required for downstream tracking
            num_joints=metadata.num_bodyparts + metadata.num_unique_bodyparts,
            all_joints=[[i] for i in range(metadata.num_bodyparts + metadata.num_unique_bodyparts)],
            all_joints_names=metadata.bodyparts + metadata.unique_bodyparts,
            net_type=cfg.net_type,
            global_scale=global_scale,
            scoremap_dir=scoremap_dir,
        )

        # Save if needed
        if save:
            if test_config_path is None:
                raise ValueError("test_config_path is required to save the test config.")
            test_config.to_yaml(test_config_path, overwrite=True)

        return test_config
