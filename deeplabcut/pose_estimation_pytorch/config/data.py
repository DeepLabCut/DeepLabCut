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
"""Data configuration classes for DeepLabCut pose estimation models."""

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from deeplabcut.core.config import DLCBaseConfig
from deeplabcut.core.config.validation import Fraction, NonNegativeFloat, NonNegativeInt


class DataLoaderType(str, Enum):
    DLCLoader = "DLCLoader"
    COCOLoader = "COCOLoader"


class DLCLoaderConfig(DLCBaseConfig):
    """Configuration for DeepLabCut Loader.

    Attributes:
        type: Loader type identifier
        config: Path to the DeepLabCut project config, or the project config itself
        trainset_index: Index of the TrainingsetFraction for which to load data
        shuffle: Index of the shuffle for which to load data
        modelprefix: The modelprefix for the shuffle
    """

    type: Literal[DataLoaderType.DLCLoader]
    config: str | dict
    trainset_index: NonNegativeInt = 0
    shuffle: NonNegativeInt = 0
    modelprefix: str = ""


class COCOLoaderConfig(DLCBaseConfig):
    """Configuration for COCO Loader.

    Attributes:
        type: Loader type identifier
    """

    type: Literal[DataLoaderType.COCOLoader]
    # TODO @deruyter92 2026-06-05: COCOLoader is never build from config, add config when needed


class DataTransformationConfig(DLCBaseConfig):
    """Data transformation configuration.

    Attributes:
        resize: Resize transformation configuration
        longest_max_size: Maximum size for longest edge
        hflip: Horizontal flip configuration
        affine: Affine transformation configuration
        random_bbox_transform: Random bbox transformation configuration
        crop_sampling: Crop sampling configuration
        hist_eq: Whether to apply histogram equalization
        motion_blur: Whether to apply motion blur
        covering: Covering/CoarseDropout transformation configuration
        elastic_transform: Elastic transformation configuration
        grayscale: Grayscale transformation configuration
        gaussian_noise: Gaussian noise standard deviation
        auto_padding: Auto padding configuration
        normalize_images: Whether to normalize images
        scale_to_unit_range: Whether to scale images to [0, 1] range
        top_down_crop: Top-down crop configuration
        collate: Collate function configuration
    """

    resize: dict | None = None
    longest_max_size: int | dict | None = None
    hflip: bool | float | dict | None = None
    affine: dict | None = None
    random_bbox_transform: dict | None = None
    crop_sampling: dict | None = None
    hist_eq: bool | dict | None = False
    motion_blur: bool | dict | None = False
    covering: bool | dict | None = None
    elastic_transform: bool | dict | None = None
    grayscale: bool | dict | None = None
    gaussian_noise: bool | float | int | dict | None = None
    auto_padding: dict | None = None
    normalize_images: bool | dict | None = True
    scale_to_unit_range: bool | dict | None = False
    top_down_crop: dict | None = None
    collate: dict | None = None


class GenSamplingConfig(DLCBaseConfig):
    """Configuration for CTD models.

    Args:
        keypoint_sigmas: The sigma for each keypoint.
        keypoints_symmetry: Indices of symmetric keypoints (e.g. left/right eye)
        jitter_prob: The probability of applying jitter. Jitter error is defined as
            a small displacement from the GT keypoint.
        swap_prob: The probability of applying a swap error. Swap error represents
            a confusion between the same or similar parts which belong to different
            persons.
        inv_prob: The probability of applying an inversion error. Inversion error
            occurs when a pose estimation model is confused between semantically
            similar parts that belong to the same instance.
        miss_prob: The probability of applying a miss error. Miss error represents a
            large displacement from the GT keypoint position.
    """

    model_config = ConfigDict(frozen=True)

    keypoint_sigmas: NonNegativeFloat | list[NonNegativeFloat] = 0.1
    keypoints_symmetry: list[tuple[int, int]] | None = None
    jitter_prob: Fraction = 0.16
    swap_prob: Fraction = 0.08
    inv_prob: Fraction = 0.03
    miss_prob: Fraction = 0.10


class DetectorDataConfig(DLCBaseConfig):
    """Data configuration for object-detector training (no pose-only fields)."""

    colormode: Literal["RGB"] = "RGB"
    inference: DataTransformationConfig | None = None
    train: DataTransformationConfig | None = None

    @field_validator("train", "inference", mode="before")
    @classmethod
    def validate_transforms(cls, v):
        from deeplabcut.pose_estimation_pytorch.data import build_transforms

        try:
            build_transforms(v)
        except Exception as e:
            raise ValueError(f"Could not build transforms. Please check your config. Config: {v}; Error: {e}") from e
        return v


class DataConfig(DLCBaseConfig):
    """Complete data configuration.

    Attributes:
        bbox_margin: Bounding box margin for top-down models
        colormode: Color mode for images (e.g., 'RGB', 'BGR')
        gen_sampling: Generation sampling configuration
        inference: Inference data configuration
        train: Training data configuration
        loader: Data loader configuration
    """

    bbox_margin: NonNegativeInt = 25
    colormode: Literal["RGB"] = "RGB"  # Docs state that it should never be changed to BGR
    gen_sampling: GenSamplingConfig | None = None
    inference: DataTransformationConfig | None = None
    train: DataTransformationConfig | None = None
    loader: DLCLoaderConfig | COCOLoaderConfig | None = Field(default=None, discriminator="type")

    @field_validator("train", "inference", mode="before")
    @classmethod
    def validate_transforms(cls, v):
        from deeplabcut.pose_estimation_pytorch.data import build_transforms

        try:
            build_transforms(v)
        except Exception as e:
            raise ValueError(f"Could not build transforms. Please check your config. Config: {v}; Error: {e}") from e
        return v
