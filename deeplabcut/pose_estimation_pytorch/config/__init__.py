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
# For backwards compatibility
from deeplabcut.core.config import (
    pretty_print,
    read_config_as_dict,
    write_config,
)
from deeplabcut.pose_estimation_pytorch.config.data import (
    COCOLoaderConfig,
    DataConfig,
    DataLoaderType,
    DataTransformationConfig,
    DLCLoaderConfig,
    GenSamplingConfig,
)
from deeplabcut.pose_estimation_pytorch.config.inference import (
    AutocastConfig,
    CompileConfig,
    EvaluationConfig,
    InferenceConfig,
    MultithreadingConfig,
)
from deeplabcut.pose_estimation_pytorch.config.logger import (
    CSVLoggerConfig,
    LoggerConfig,
    LoggerType,
    WandbLoggerConfig,
)
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import (
    add_detector,
    add_identity_head,
    add_metadata,
    add_unique_bodypart_head,
    create_backbone_with_heatmap_model,
    create_backbone_with_paf_model,
    make_basic_project_config,
    make_pytorch_pose_config,
    make_pytorch_test_config,
)
from deeplabcut.pose_estimation_pytorch.config.model import (
    DetectorModelConfig,
    ModelConfig,
)
from deeplabcut.pose_estimation_pytorch.config.pose import (
    DatasetType,
    DetectorConfig,
    MethodType,
    NetType,
    PoseConfig,
    PoseMetadata,
    TestConfig,
)
from deeplabcut.pose_estimation_pytorch.config.runner import (
    OptimizerConfig,
    RunnerConfig,
    SchedulerConfig,
    SnapshotCheckpointConfig,
)
from deeplabcut.pose_estimation_pytorch.config.training import TrainSettingsConfig
from deeplabcut.pose_estimation_pytorch.config.utils import (
    available_detectors,
    available_models,
    get_config_folder_path,
    is_model_cond_top_down,
    is_model_top_down,
    load_backbones,
    load_base_config,
    load_detectors,
    replace_default_values,
    update_config,
    update_config_by_dotpath,
)

__all__ = [
    # Backwards compatibility (core config I/O)
    "pretty_print",
    "read_config_as_dict",
    "write_config",
    # Config creation API
    "add_detector",
    "add_identity_head",
    "add_metadata",
    "add_unique_bodypart_head",
    "create_backbone_with_heatmap_model",
    "create_backbone_with_paf_model",
    "make_basic_project_config",
    "make_pytorch_pose_config",
    "make_pytorch_test_config",
    # Config utilities
    "available_detectors",
    "available_models",
    "get_config_folder_path",
    "is_model_cond_top_down",
    "is_model_top_down",
    "load_backbones",
    "load_base_config",
    "load_detectors",
    "replace_default_values",
    "update_config",
    "update_config_by_dotpath",
    # Data config
    "COCOLoaderConfig",
    "DataConfig",
    "DataLoaderType",
    "DataTransformationConfig",
    "DLCLoaderConfig",
    "GenSamplingConfig",
    # Inference config
    "AutocastConfig",
    "CompileConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "MultithreadingConfig",
    # Logger config
    "CSVLoggerConfig",
    "LoggerConfig",
    "LoggerType",
    "WandbLoggerConfig",
    # Model config
    "DetectorModelConfig",
    "ModelConfig",
    # Pose config
    "DatasetType",
    "DetectorConfig",
    "MethodType",
    "NetType",
    "PoseConfig",
    "PoseMetadata",
    "TestConfig",
    # Runner config
    "OptimizerConfig",
    "RunnerConfig",
    "SchedulerConfig",
    "SnapshotCheckpointConfig",
    # Training config
    "TrainSettingsConfig",
]
