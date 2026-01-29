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
"""Project configuration classes for DeepLabCut pose estimation models."""

from typing import Any

from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class ProjectConfig(ConfigMixin):
    """Complete project configuration.

    Mirrors the structure of the project config.yaml (and metadata in pose config).
    Field names match the old dictionary keys for round-trip compatibility.

    Attributes:
        Task: Project task identifier (do not edit).
        scorer: Scorer name (do not edit).
        date: Project date (do not edit).
        multianimalproject: Whether the project is multi-animal.
        identity: Whether identity tracking is enabled (project config.yaml key).
        project_path: Path to the DeepLabCut project.
        pose_config_path: Path to the pose configuration file (metadata only).
        engine: Default DeepLabCut engine (e.g. pytorch).
        video_sets: Video set configuration.
        bodyparts: List of body parts.
        individuals: List of individual animal identities (multi-animal).
        uniquebodyparts: List of unique body parts (multi-animal project key).
        multianimalbodyparts: List of multi-animal body parts (multi-animal key).
        unique_bodyparts: List of unique body parts (metadata key).
        start: Fraction of video to start extracting frames.
        stop: Fraction of video to stop extracting frames.
        numframes2pick: Number of frames to pick for labeling.
        skeleton: Skeleton connectivity for plotting.
        skeleton_color: Skeleton color for plotting.
        pcutoff: Confidence cutoff for plotting.
        dotsize: Dot size for visualization.
        alphavalue: Alpha value for visualization.
        colormap: Colormap for visualization.
        TrainingFraction: Training fractions for dataset splits.
        iteration: Training iteration.
        default_net_type: Default network architecture.
        default_augmenter: Default data augmenter.
        default_track_method: Default tracking method.
        snapshotindex: Snapshot index for evaluation.
        detector_snapshotindex: Detector snapshot index.
        batch_size: Training batch size.
        detector_batch_size: Detector batch size.
        cropping: Whether cropping is enabled for analysis.
        x1: Cropping x1 coordinate.
        x2: Cropping x2 coordinate.
        y1: Cropping y1 coordinate.
        y2: Cropping y2 coordinate.
        corner2move2: Refinement corner configuration.
        move2corner: Refinement move-to-corner flag.
        SuperAnimalConversionTables: Conversion tables for SuperAnimal weights.
    """

    # Project definitions (do not edit) - from config.yaml template
    Task: str = ""
    scorer: str = ""
    date: str = ""
    multianimalproject: bool = False
    identity: bool | None = None

    # Project path
    project_path: str = ""
    pose_config_path: str = ""  # used in pose config metadata only

    # Engine
    engine: str = "pytorch"

    # Annotation data set configuration
    video_sets: dict[str, Any] = field(default_factory=dict)
    # VV TODO @deruyter92 2026-01-30: following the old original config.yaml template for now. VV
    # VV We should change this to a list[str] in the future. VV
    bodyparts: list[str] | str = 'MULTI!' 
    individuals: list[str] = field(default_factory=list)
    uniquebodyparts: list[str] = field(default_factory=list)  # multi-animal project key
    multianimalbodyparts: list[str] = field(
        default_factory=list
    )  # multi-animal project key
    unique_bodyparts: list[str] = field(
        default_factory=list
    )  # metadata key; same as uniquebodyparts

    # Fraction of video to start/stop when extracting frames
    start: float = 0.0
    stop: float = 1.0
    numframes2pick: int = 20

    # Plotting configuration
    skeleton: list[list[str]] = field(default_factory=list)
    skeleton_color: str = "black"
    pcutoff: float = 0.4
    dotsize: int = 12
    alphavalue: float = 0.7
    colormap: str = "rainbow"

    # Training, evaluation and analysis configuration
    TrainingFraction: list[float] = field(default_factory=list)
    iteration: int | None = None
    default_net_type: str = "resnet_50"
    default_augmenter: str | None = None
    default_track_method: str | None = None
    snapshotindex: str | int = "all"
    detector_snapshotindex: int = -1
    batch_size: int = 8
    detector_batch_size: int = 1

    # Cropping parameters (for analysis and outlier frame detection)
    cropping: bool = False
    x1: int | None = None
    x2: int | None = None
    y1: int | None = None
    y2: int | None = None

    # Refinement configuration
    corner2move2: list[list[str]] | None = None
    move2corner: bool | None = None

    # Conversion tables to fine-tune SuperAnimal weights
    SuperAnimalConversionTables: dict[str, Any] | None = None
