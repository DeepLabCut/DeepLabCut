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
"""Inference configuration classes for DeepLabCut pose estimation models."""

from typing import Literal
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class MultithreadingConfig:
    """Multithreading configuration for inference.

    Attributes:
        enabled: Whether multithreading is enabled
        queue_length: Length of the processing queue
        timeout: Timeout for processing tasks
    """

    enabled: bool = False
    queue_length: int = 4
    timeout: float = 30.0


@dataclass
class CompileConfig:
    """Model compilation configuration for inference optimization.

    Attributes:
        enabled: Whether compilation is enabled
        mode: Compilation mode
    """

    enabled: bool = False
    backend: str = "inductor"


@dataclass
class AutocastConfig:
    """Automatic mixed precision configuration.

    Attributes:
        enabled: Whether autocast is enabled
        dtype: Data type for autocast (float16, bfloat16)
    """

    enabled: bool = False

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics computation.
    
    Attributes:
        pcutoff: Confidence threshold for RMSE computation. Can be:
            - float: Single threshold for all bodyparts
            - list[float]: One value per bodypart (and unique bodypart if any)
            - dict[str, float]: Mapping bodypart names to thresholds
        comparison_bodyparts: Subset of bodyparts to compute metrics for.
            Can be "all", None (all bodyparts), or a list of bodypart names.
        per_keypoint_evaluation: Whether to compute train and test RMSE 
            for each keypoint individually.
        force_multi_animal: If True, use multi-animal evaluation even if 
            loader contains only a single animal.
    """
    mode: Literal["train", "test", "all"] = "all"
    pcutoff: float | list[float] | dict[str, float] = 0.6
    comparison_bodyparts: Literal["all"] | list[str] | None = "all"
    per_keypoint_evaluation: bool = False
    force_multi_animal: bool = False

@dataclass
class InferenceConfig:
    """Complete inference configuration.

    Attributes:
        multithreading: Multithreading configuration
        compile: Compilation configuration
        autocast: Autocast configuration
        conditions: Conditions for conditional models (CTD)
        snapshot: Snapshot(s) to use for inference
        eval: Evaluation configuration
    """

    multithreading: MultithreadingConfig = field(default_factory=MultithreadingConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    autocast: AutocastConfig = field(default_factory=AutocastConfig)
    conditions: dict[str, Any] | None = None
    snapshot: int | str | list[int] | None = None
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str | None = None
