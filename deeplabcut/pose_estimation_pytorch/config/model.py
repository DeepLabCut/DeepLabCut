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
"""Model configuration class for DeepLabCut pose estimation models."""

from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class ModelConfig:
    """Complete model configuration.

    Attributes:
        backbone: Backbone configuration
        backbone_output_channels: Number of output channels from backbone
        heads: Dictionary of head configurations by name
        neck: Neck configuration
        pose_model: Pose model configuration
    """

    backbone: dict = field(default_factory=dict)
    heads: dict[str, dict] = field(default_factory=dict)
    backbone_output_channels: int | None = None
    neck: dict | None = None
    pose_model: dict | None = None


@dataclass
class DetectorModelConfig:
    """Configuration for detector models

    Attributes:
        type: Type of detector model (e.g., FasterRCNN)
        freeze_bn_stats: Whether to freeze batch normalization statistics
        freeze_bn_weights: Whether to freeze batch normalization weights
        variant: Specific variant of the detector model
    """

    type: str = ""
    freeze_bn_stats: bool = False
    freeze_bn_weights: bool = False
    variant: str = ""
