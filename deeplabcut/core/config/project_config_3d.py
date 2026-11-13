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
"""Project configuration of a 3D (multi-camera) DeepLabCut project."""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, model_serializer, model_validator
from ruamel.yaml.comments import CommentedMap

from deeplabcut.core.config.base_config import DLCBaseConfig, DLCVersionedConfig
from deeplabcut.core.config.project_config import repair_project_path
from deeplabcut.core.config.validation import BodypartPair, Fraction, NonNegativeInt, StrictPositiveInt

logger = logging.getLogger(__name__)

# DeepLabCut <= 3.0.0 read 3D configs with the 2D read_config, which could write them back
# with the keys of the 2D project template. They have no meaning in a 3D project.
# fmt: off
_LEGACY_2D_KEYS = frozenset({
    "multianimalproject", "identity", "pose_config_path", "engine", "video_sets", "bodyparts",
    "individuals", "uniquebodyparts", "multianimalbodyparts", "start", "stop", "numframes2pick",
    "alphavalue", "TrainingFraction", "iteration", "default_net_type", "default_augmenter",
    "default_track_method", "snapshotindex", "detector_snapshotindex", "batch_size",
    "detector_batch_size", "cropping", "x1", "x2", "y1", "y2", "corner2move2", "move2corner",
    "SuperAnimalConversionTables", "resnet", "croppedtraining",
})
# fmt: on


class CameraConfig3D(DLCBaseConfig):
    """Settings of one camera of a 3D project.

    Attributes:
        config_file: Path to the config.yaml of the 2D project used to analyze this camera's videos.
        shuffle: Shuffle of that 2D project.
        trainingsetindex: Training set index of that 2D project.
    """

    config_file: Path | None = None
    shuffle: NonNegativeInt = 1
    trainingsetindex: NonNegativeInt = 0


class ProjectConfig3D(DLCVersionedConfig):
    """Complete configuration of a 3D project (the 3D project's config.yaml).

    In config.yaml the cameras are stored as a ``camera_names`` list plus flat
    ``<setting>_<camera name>`` keys (e.g. ``shuffle_camera-1``). The model groups them
    into `cameras` (camera name -> settings) and writes them back in the flat layout, so the
    file stays readable by older DeepLabCut versions.

    Attributes:
        Task: Project task identifier (do not edit).
        scorer: Scorer name (do not edit).
        date: Project date (do not edit).
        project_path: Path to the 3D project.
        skeleton: Skeleton connectivity for plotting.
        skeleton_color: Skeleton color for plotting.
        pcutoff: Confidence cutoff for plotting and triangulation.
        colormap: Colormap for visualization.
        dotsize: Dot size for visualization.
        alphaValue: Alpha value for visualization.
        markerType: Marker type for visualization.
        markerColor: Marker color for visualization.
        num_cameras: Number of cameras.
        cameras: Settings per camera, keyed by camera name (in camera order).
        scorername_3d: Scorer name for the 3D output.
    """

    # Project definitions (do not edit)
    Task: str = Field(default="", json_schema_extra={"comment": "Project definitions (do not edit)"})
    scorer: str = ""
    date: str = ""

    # Project path
    project_path: Path = Field(
        default_factory=Path,
        json_schema_extra={"comment": "\nProject path (change when moving around)"},
    )

    # Plotting configuration
    skeleton: list[BodypartPair] = Field(
        default_factory=list,
        json_schema_extra={"comment": "\nPlotting configuration"},
    )
    skeleton_color: str = "black"
    pcutoff: Fraction = 0.4
    colormap: str = "jet"
    dotsize: NonNegativeInt = 15
    alphaValue: Fraction = 0.8
    markerType: str = "*"
    markerColor: str = "r"

    # Cameras
    num_cameras: StrictPositiveInt = Field(
        default=2,
        json_schema_extra={
            "comment": (
                "\nNumber of cameras, camera names, path of the config files, shuffle index and "
                "trainingsetindex used to analyze videos:"
            )
        },
    )
    cameras: dict[str, CameraConfig3D] = Field(min_length=1)
    scorername_3d: str = "DLC_3D"

    @property
    def camera_names(self) -> list[str]:
        """Names of the cameras, used to match videos and calibration images."""
        return list(self.cameras)

    def _post_yaml_load_updates(self, *, yaml_path: Path) -> None:
        repair_project_path(self, yaml_path=yaml_path)

    # ------------------------------------------------------------------
    # config.yaml layout: `camera_names` + flat `<setting>_<camera>` keys  <->  `cameras`
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def group_flat_camera_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "camera_names" not in data:
            return data  # already grouped (constructor kwargs, assignment)
        if ignored := sorted(_LEGACY_2D_KEYS & data.keys()):
            logger.info(f"Ignoring 2D project keys in the 3D project config: {ignored}")
        data = {key: value for key, value in data.items() if key not in _LEGACY_2D_KEYS}

        cameras = {}
        for i, name in enumerate(data.pop("camera_names"), start=1):
            settings = {}
            for setting in CameraConfig3D.model_fields:
                # Keys renamed to the camera name take precedence over the template's camera-<i> keys
                # (calibrate_cameras used to rename only some of them).
                template_value = data.pop(f"{setting}_camera-{i}", None)
                value = data.pop(f"{setting}_{name}", template_value)
                if value is not None:
                    settings[setting] = value
            cameras[name] = settings
        data["cameras"] = cameras
        return data  # keys of unknown cameras are left for pydantic to reject

    @model_serializer(mode="wrap")
    def flatten_camera_keys(self, handler) -> dict[str, Any]:
        data = handler(self)
        if (cameras := data.pop("cameras", None)) is None:
            return data  # excluded from the dump
        data["camera_names"] = list(cameras)
        for setting in CameraConfig3D.model_fields:
            for name, settings in cameras.items():
                data[f"{setting}_{name}"] = settings[setting]
        return data

    def to_commented_map(self) -> CommentedMap:
        data = super().to_commented_map()  # declared fields only; `cameras` is flattened by the serializer
        for key, value in self.to_dict(normalize=True).items():
            data.setdefault(key, value)
        return data
