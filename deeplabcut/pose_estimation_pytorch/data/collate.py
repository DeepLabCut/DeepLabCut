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
"""Custom collate functions"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import default_collate

from deeplabcut.pose_estimation_pytorch.data.image import resize_and_random_crop
from deeplabcut.pose_estimation_pytorch.registry import build_from_cfg, Registry


COLLATE_FUNCTIONS = Registry("collate_functions", build_func=build_from_cfg)


class CollateFunction(ABC):
    """A class that can be called as a collate function"""

    @abstractmethod
    def __call__(self, batch) -> dict | list:
        """Returns: the collated batch"""
        raise NotImplementedError()


class ResizeCollate(CollateFunction, ABC):
    """A collate function which resizes all images in a batch to the same size

    Args:
        max_shift: The maximum shift, in pixels, to add to the random crop (this means
            there can be a slight border around the image)
        max_size: The maximum size of the long edge of the image when resized. If the
            longest side will be greater than this value, resizes such that the longest
            side is this size, and the shortest side is smaller than the desired size.
            This is useful to keep some information from images with extreme aspect
            ratios.
        seed: The random seed to use to sample scales/sizes.
    """

    def __init__(
        self,
        max_shift: int = 10,
        max_size: int = 2048,
        seed: int = 0,
    ) -> None:
        self.generator = np.random.default_rng(seed=seed)
        self.max_size = max_size
        self.max_shift = max_shift
        self._current_batch = []

    @abstractmethod
    def _sample_scale(self) -> int | tuple[int, int]:
        """Returns: the target shape for images in the batch"""
        raise NotImplementedError()

    def __call__(self, batch) -> dict | list:
        """Returns: the collated batch"""
        self._current_batch = batch
        new_size = self._sample_scale()
        updated_batch = []
        for item in batch:
            image, new_targets = resize_and_random_crop(
                image=item["image"],
                targets=item,
                size=new_size,
                max_size=self.max_size,
                max_shift=self.max_shift,
            )
            new_targets["image"] = image
            updated_batch.append(new_targets)

        return default_collate(updated_batch)


@COLLATE_FUNCTIONS.register_module
class ResizeFromDataSizeCollate(ResizeCollate):
    """A collate function which resizes all images in a batch to the same size

    The target size is obtained by taking the size of the first image in the batch, and
    multiplying it by a scale taken uniformly at random from (min_scale, max_scale).

    The aspect ratio of all images in the batch is preserved, with cropping/padding used
    to generate images of the correct shapes.

    If to_square:
        The images will be resized to squares, where the side is the short side of the
        original image.
    else:
        The images will be resized to a scaled version of the shape of the first image.

    Args:
        min_scale: The minimum scale factor to apply to the image size
        max_scale: The maximum scale factor to apply to the image size
        min_short_side: The smallest size for the target short side.
        max_short_side: The largest size for the target short side.
        max_ratio: The largest aspect ratio allowed for a target (longSide / shortSide).
            If the aspect ratio is larger, it will be clamped to max_ratio. Must be >=1.
        multiple_of: If defined, the height and width of all target sizes will be a
            multiple of this value.
        to_square: Whether images should be resized to squares.
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        min_short_side: int = 128,
        max_short_side: int = 1152,
        max_ratio: float = 2.0,
        multiple_of: int | None = None,
        to_square: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_short_side = min_short_side
        self.max_short_side = max_short_side
        self.max_ratio = max_ratio
        self.multiple_of = multiple_of
        self.to_square = to_square

    def _sample_scale(self) -> int | tuple[int, int]:
        if len(self._current_batch) == 0:
            raise ValueError("Cannot sample frame shape: no items in current batch")

        h, w = self._current_batch[0]["image"].shape[1:]
        scale = self.generator.uniform(self.min_scale, self.max_scale)
        if self.to_square:
            short_side = min(h, w)
            size = int(round(
                min(self.max_short_side, max(self.min_short_side, scale * short_side))
            ))
            if self.multiple_of is not None:
                size = _to_multiple(size, self.multiple_of)
            return size

        short, long = min(h, w), max(h, w)
        ratio = long / short
        if ratio > self.max_ratio:
            ratio = self.max_ratio

        short_size = int(
            round(min(self.max_short_side, max(self.min_short_side, scale * short)))
        )
        if h < w:
            h = short_size
            w = int(ratio * short_size)
        else:
            h = int(ratio * short_size)
            w = short_size

        if self.multiple_of is not None:
            w = _to_multiple(w, self.multiple_of)
            h = _to_multiple(h, self.multiple_of)

        return h, w


@COLLATE_FUNCTIONS.register_module
class ResizeFromListCollate(ResizeCollate):
    """A collate function which resizes all images in a batch to the same size

    The target size image size is sampled from a list. If it's a list of integers,
    all images will be resized into squares. If it's a list of tuples, that will be the
    target (h, w) for images.

    Args:
        scales: The target sizes to resize the images to.
    """

    def __init__(self, scales: list[int] | list[tuple[int, int]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.scales = scales

    def _sample_scale(self) -> int | tuple[int, int]:
        return self.generator.choice(self.scales)


def _to_multiple(value: int, of: int) -> int:
    """Returns: the smallest integer >= ``value`` which is a multiple of ``of``"""
    return of * ((value + of - 1) // of)
