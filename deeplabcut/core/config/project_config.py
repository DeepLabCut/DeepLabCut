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

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from deeplabcut.core.config.base_config import DLCVersionedConfig
from deeplabcut.core.config.validation import (
    Fraction,
    NonNegativeInt,
    StrictPositiveInt,
    UniqueStrList,
    less_than,
)


class ProjectConfig(DLCVersionedConfig):
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

    # Project definitions (do not edit)
    Task: str = Field(default="", json_schema_extra={"comment": "Project definitions (do not edit)"})
    scorer: str = ""
    date: str = ""
    multianimalproject: bool = False
    identity: bool | None = Field(default=None, json_schema_extra={"aliases": ["with_identity"]})

    # Project path
    project_path: Path = Field(
        default_factory=Path,
        json_schema_extra={"comment": "\nProject path (change when moving around)"},
    )
    pose_config_path: Path = Path()

    # Engine
    engine: Literal["pytorch", "tensorflow"] = Field(
        default="pytorch",
        json_schema_extra={
            "comment": "\nDefault DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)"
        },
    )

    # Annotation dataset configuration (and individual video cropping parameters)
    video_sets: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"comment": "\nAnnotation data set configuration (and individual video cropping parameters)"},
    )
    # VV TODO @deruyter92 2026-01-30: following the old original config.yaml template for now. VV
    # VV We should change this to a list[str] in the future. VV
    bodyparts: UniqueStrList | str = "MULTI!"

    # TODO @deruyter92 2026-02-06: The current pipeline requires at least one individual defined in the
    # default configuration. This will be removed in the future.
    individuals: UniqueStrList = Field(default_factory=lambda: ["individual_1"])
    uniquebodyparts: UniqueStrList = Field(default_factory=list, json_schema_extra={"aliases": ["unique_bodyparts"]})
    multianimalbodyparts: UniqueStrList = Field(default_factory=list)  # multi-animal project key

    # Fraction of video to start/stop when extracting frames for labeling/refinement
    start: Fraction = Field(
        default=0.0,
        json_schema_extra={
            "comment": "\nFraction of video to start/stop when extracting frames for labeling/refinement"
        },
    )
    stop: Fraction = 1.0
    numframes2pick: NonNegativeInt = 20

    # Plotting configuration
    skeleton: list[UniqueStrList] = Field(
        default_factory=list,
        json_schema_extra={"comment": "\nPlotting configuration"},
    )
    skeleton_color: str = "black"
    pcutoff: Fraction = 0.4
    dotsize: NonNegativeInt = 12
    alphavalue: Fraction = 0.7
    colormap: str = "rainbow"

    # Training, evaluation and analysis configuration
    TrainingFraction: list[Fraction] = Field(
        default_factory=list,
        json_schema_extra={"comment": "\nTraining,Evaluation and Analysis configuration"},
    )
    iteration: NonNegativeInt | None = None
    default_net_type: str = "resnet_50"
    default_augmenter: str | None = None
    default_track_method: str | None = None
    snapshotindex: str | int = "all"
    detector_snapshotindex: int = -1
    batch_size: StrictPositiveInt = 8
    detector_batch_size: StrictPositiveInt = 1

    # Cropping parameters (for analysis and outlier frame detection)
    cropping: bool = Field(
        default=False,
        json_schema_extra={"comment": "\nCropping Parameters (for analysis and outlier frame detection)"},
    )
    x1: NonNegativeInt | None = Field(
        default=None,
        json_schema_extra={"comment": "if cropping is true for analysis, then set the values here:"},
    )
    x2: NonNegativeInt | None = None
    y1: NonNegativeInt | None = None
    y2: NonNegativeInt | None = None

    # Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
    corner2move2: list[NonNegativeInt] | None = Field(
        default=None,
        json_schema_extra={
            "comment": (
                "\nRefinement configuration (parameters from annotation dataset "
                "configuration also relevant in this stage)"
            )
        },
    )
    move2corner: bool | None = None

    # Conversion tables to fine-tune SuperAnimal weights
    SuperAnimalConversionTables: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={"comment": "\nConversion tables to fine-tune SuperAnimal weights"},
    )

    def _post_yaml_load_updates(self, *, yaml_path: Path) -> None:
        """
        Override method for post-yaml load updates. Called automatically by from_yaml().
        These are logged but not written to disk -- call to_yaml() explicitly if needed.
        """
        project_path = yaml_path.parent
        if project_path.absolute() != self.project_path.absolute():
            old = self.project_path
            self.project_path = project_path
            self.record_change_note(
                "project_path",
                f"project_path updated: {old} -> {project_path} (resolved from YAML location when reading config.yaml)",
            )

    @model_validator(mode="after")
    def validate_start_before_stop(self) -> Self:
        less_than(self.start, self.stop, name="start", threshold_name="stop")
        return self
