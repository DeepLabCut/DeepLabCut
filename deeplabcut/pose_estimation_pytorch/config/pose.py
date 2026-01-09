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

from pydantic.dataclasses import dataclass
from dataclasses import field
from pydantic import Field
from enum import Enum

from deeplabcut.pose_estimation_pytorch.config.project import ProjectConfig
from deeplabcut.pose_estimation_pytorch.config.data import DataConfig
from deeplabcut.pose_estimation_pytorch.config.training import TrainSettingsConfig
from deeplabcut.pose_estimation_pytorch.config.runner import RunnerConfig
from deeplabcut.pose_estimation_pytorch.config.inference import InferenceConfig
from deeplabcut.pose_estimation_pytorch.config.model import ModelConfig, DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.config.logger import CSVLoggerConfig, WandbLoggerConfig


class MethodType(str, Enum):
    """Enumeration of pose estimation method types."""

    BOTTOM_UP = "bu"
    TOP_DOWN = "td"
    CONDITIONAL_TOP_DOWN = "ctd"


class NetType(str, Enum):
    """Enumeration of network architecture types."""

    # ResNet variants (bottom-up)
    RESNET_50 = "resnet_50"
    RESNET_101 = "resnet_101"

    # ResNet variants (top-down)
    TOP_DOWN_RESNET_50 = "top_down_resnet_50"
    TOP_DOWN_RESNET_101 = "top_down_resnet_101"

    # HRNet variants (bottom-up)
    HRNET_W18 = "hrnet_w18"
    HRNET_W32 = "hrnet_w32"
    HRNET_W48 = "hrnet_w48"

    # HRNet variants (top-down)
    TOP_DOWN_HRNET_W18 = "top_down_hrnet_w18"
    TOP_DOWN_HRNET_W32 = "top_down_hrnet_w32"
    TOP_DOWN_HRNET_W48 = "top_down_hrnet_w48"

    # DEKR variants (bottom-up with HRNet backbone)
    DEKR_W18 = "dekr_w18"
    DEKR_W32 = "dekr_w32"
    DEKR_W48 = "dekr_w48"

    # BUCTD variants (Conditional Top-Down)
    CTD_COAM_W32 = "ctd_coam_w32"
    CTD_COAM_W48 = "ctd_coam_w48"
    CTD_COAM_W48_HUMAN = "ctd_coam_w48_human"
    CTD_PRENET_HRNET_W32 = "ctd_prenet_hrnet_w32"
    CTD_PRENET_HRNET_W48 = "ctd_prenet_hrnet_w48"
    CTD_PRENET_RTMPOSE_S = "ctd_prenet_rtmpose_s"
    CTD_PRENET_RTMPOSE_M = "ctd_prenet_rtmpose_m"
    CTD_PRENET_RTMPOSE_X = "ctd_prenet_rtmpose_x"
    CTD_PRENET_RTMPOSE_X_HUMAN = "ctd_prenet_rtmpose_x_human"

    # DLCRNet variants
    DLCRNET_STRIDE16_MS5 = "dlcrnet_stride16_ms5"
    DLCRNET_STRIDE32_MS5 = "dlcrnet_stride32_ms5"

    # RTMPose variants (top-down)
    RTMPOSE_S = "rtmpose_s"
    RTMPOSE_M = "rtmpose_m"
    RTMPOSE_X = "rtmpose_x"

    # AnimalTokenPose variant (inference only)
    ANIMALTOKENPOSE_BASE = "animaltokenpose_base"


@dataclass
class DetectorConfig:
    model: DetectorModelConfig
    device: str = "auto"
    data: DataConfig | None = None
    runner: RunnerConfig | None = None
    train_settings: TrainSettingsConfig | None = None
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass
class PoseConfig:
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

    version: str = "1.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    net_type: NetType = NetType.RESNET_50
    method: MethodType = MethodType.BOTTOM_UP
    device: str = "auto"
    metadata: ProjectConfig | None = None
    data: DataConfig | None = None
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logger: CSVLoggerConfig | WandbLoggerConfig | None = Field(default=None, discriminator="type")
    with_center_keypoints: bool = False
    runner: RunnerConfig | None = None
    train_settings: TrainSettingsConfig | None = None
    detector: DetectorConfig | None = None
