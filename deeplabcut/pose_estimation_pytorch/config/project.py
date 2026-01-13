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

from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class ProjectConfig:
    """Complete project configuration.

    Attributes:
        project_path: Path to the DeepLabCut project
        pose_config_path: Path to the pose configuration file
        bodyparts: List of body parts
        unique_bodyparts: List of unique body parts
        individuals: List of individual animal identities
        with_identity: Whether identity tracking is enabled
        multianimalproject: Whether the project is a multi-animal project
        colormap: Colormap for visualization
        dotsize: Dot size for visualization
        alphavalue: Alpha value for visualization
    """

    project_path: str = ""
    pose_config_path: str = ""
    bodyparts: list[str] = field(default_factory=list)
    unique_bodyparts: list[str] = field(default_factory=list)
    individuals: list[str] = field(default_factory=list)
    with_identity: bool | None = None
    colormap: str = "rainbow"
    dotsize: int = 12
    alphavalue: float = 0.7