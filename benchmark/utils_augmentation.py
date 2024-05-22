from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchCollate:
    """Resize + scale images when batching"""
    min_scale: float
    max_scale: float
    min_short_side: int = 256
    max_short_side: int = 1152
    multiple_of: int | None = None
    max_ratio: float = 2.0
    to_square: bool = False

    def data(self) -> dict:
        return {
            "type": "ResizeFromDataSizeCollate",
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "min_short_side": self.min_short_side,
            "max_short_side": self.max_short_side,
            "max_ratio": self.max_ratio,
            "multiple_of": self.multiple_of,
            "to_square": self.to_square,
        }


@dataclass
class AutoPadding:
    """Random crop around keypoints"""
    pad_height_divisor: int
    pad_width_divisor: int
    border_mode: str = "constant"

    def data(self) -> dict:
        return {
            "pad_height_divisor": self.pad_height_divisor,
            "pad_width_divisor": self.pad_width_divisor,
            "border_mode": self.border_mode,
        }


@dataclass
class AffineAugmentation:
    """An affine image augmentation"""

    p: float = 0.5
    rotation: int = 0
    scale: tuple[float, float] = (1, 1)
    translation: int = 0

    def data(self) -> dict:
        return {
            "p": self.p,
            "scaling": self.scale,
            "rotation": self.rotation,
            "translation": self.translation,
        }


@dataclass
class CropSampling:
    """Random crop around keypoints"""
    width: int
    height: int
    max_shift: float = 0.4
    method: str = "uniform"  # "uniform", "keypoints", "density", "hybrid"

    def __post_init__(self):
        assert self.method in ("uniform", "keypoints", "density", "hybrid")
        assert 0 <= self.max_shift <= 1

    def data(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "max_shift": self.max_shift,
            "method": self.method,
        }


@dataclass
class ImageAugmentations:
    """
    The default augmentation only normalizes images.

    Examples:
        gaussian_noise: 12.75
        resize: {height: 800, width: 800, keep_ratio: true}
        rotation: 30
        scale_jitter: (0.5, 1.25)
        translation: 40
    """

    normalize: bool = True
    affine: AffineAugmentation | None = None
    auto_padding: AutoPadding | None = None
    covering: bool = False
    gaussian_noise: float | bool = False
    hist_eq: bool = False
    motion_blur: bool = False
    hflip: bool | float = False
    resize: dict | None = None
    crop_sampling: CropSampling | None = None
    collate: BatchCollate | None = None

    def data(self) -> dict:
        augmentations = {
            "normalize_images": self.normalize,
            "covering": self.covering,
            "gaussian_noise": self.gaussian_noise,
            "hist_eq": self.hist_eq,
            "motion_blur": self.motion_blur,
            "hflip": self.hflip,
            "auto_padding": False,
            "affine": False,
            "resize": False,
            "crop_sampling": False,
            "collate": False,
        }
        if self.auto_padding is not None:
            augmentations["auto_padding"] = self.auto_padding.data()
        if self.affine is not None:
            augmentations["affine"] = self.affine.data()
        if self.resize is not None:
            augmentations["resize"] = self.resize
        if self.crop_sampling is not None:
            augmentations["crop_sampling"] = self.crop_sampling.data()
        if self.collate is not None:
            augmentations["collate"] = self.collate.data()
        return augmentations
