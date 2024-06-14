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
"""Helpers to run preprocess data before running inference"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import albumentations as A
import cv2
import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.data.utils import _crop_and_pad_image_torch

Image = TypeVar("Image", torch.Tensor, np.ndarray, str, Path)
Context = TypeVar("Context", dict[str, Any], None)


class Preprocessor(ABC):
    """
    Class to preprocess an image and turn it into a batch of inputs before running
    inference.

    As an example, a pre-processor can load an image, use a "bboxes" key from context
    to crop bounding boxes for individuals (going from a (h, w, 3) array to a
    (num_individuals, h, w, 3) array), and convert it into a tensor ready for inference.
    """

    @abstractmethod
    def __call__(self, image: Image, context: Context) -> tuple[Image, Context]:
        """Pre-processes an image

        Args:
            image: an image (containing height, width and channel dimensions) or a
                batch of images linked to a single input (containing an extra batch
                dimension)
            context: the context for this image or batch of images (such as bounding
                boxes, conditional pose, ...)

        Returns:
            the pre-processed image (or batch of images) and their context
        """
        pass


def build_bottom_up_preprocessor(
    color_mode: str, transform: A.BaseCompose
) -> Preprocessor:
    """Creates a preprocessor for bottom-up pose estimation (or object detection)

    Creates a preprocessor that loads an image, runs some transform on it (such as
    normalization), creates a tensor from the numpy array (going from (h, w, 3) to
    (3, h, w)) and adds a batch dimension (so the final tensor shape is (1, 3, h, w))

    Args:
        color_mode: whether to load the image as an RGB or BGR
        transform: the transform to apply to the image

    Returns:
        A default bottom-up Preprocessor
    """
    return ComposePreprocessor(
        components=[
            LoadImage(color_mode),
            AugmentImage(transform),
            ToTensor(),
            ToBatch(),
        ]
    )


def build_top_down_preprocessor(
    color_mode: str, transform: A.BaseCompose, cropped_image_size: tuple[int, int]
) -> Preprocessor:
    """Creates a preprocessor for top-down pose estimation

    Creates a preprocessor that loads an image, crops all bounding boxes given as a
    context (through a "bboxes" key), runs some transforms on each cropped image (such
    as normalization), creates a tensor from the numpy array (going from
    (num_ind, h, w, 3) to (num_ind, 3, h, w)).

    Args:
        color_mode: whether to load the image as an RGB or BGR
        transform: the transform to apply to the image
        cropped_image_size: the size of images for each individual to give to the pose
            estimator

    Returns:
        A default top-down Preprocessor
    """
    return ComposePreprocessor(
        components=[
            LoadImage(color_mode),
            TorchCropDetections(cropped_image_size=cropped_image_size[0]),
            AugmentImage(transform),
            ToTensor(),
        ]
    )


class ComposePreprocessor(Preprocessor):
    """
    Class to preprocess an image and turn it into a batch of
    inputs before running inference
    """

    def __init__(self, components: list[Preprocessor]) -> None:
        self.components = components

    def __call__(self, image: Image, context: Context) -> tuple[Image, Context]:
        for preprocessor in self.components:
            image, context = preprocessor(image, context)
        return image, context


class LoadImage(Preprocessor):
    """Loads an image from a file, if not yet loaded"""

    def __init__(self, color_mode: str = "RBG") -> None:
        self.color_mode = color_mode

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        if isinstance(image, (str, Path)):
            image_ = cv2.imread(str(image))
            if self.color_mode == "RGB":
                image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        else:
            image_ = image

        return image_, context


class AugmentImage(Preprocessor):
    """

    Adds an offset and scale key to the context:
        offset: (x, y) position of the pixel in the top left corner of the augmented
            image in the original image
        scale: size of the original image divided by the size of the new image

    This allows to map the position of predictions in the transformed image back to the
    original image space.
        p_original = p_transformed * scale + offset
        p_transformed = (p_original - offset) / scale
    """

    def __init__(self, transform: A.BaseCompose) -> None:
        self.transform = transform

    @staticmethod
    def get_offsets_and_scales(
        h: int,
        w: int,
        output_bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        offsets, scales = [], []
        for bbox in output_bboxes:
            x_origin, y_origin, w_out, h_out = bbox
            x_scale, y_scale = w / w_out, h / h_out
            x_offset = -x_origin * x_scale
            y_offset = -y_origin * y_scale
            offsets.append((x_offset, y_offset))
            scales.append((x_scale, y_scale))

        return offsets, scales

    @staticmethod
    def update_offset(
        offset: tuple[float, float],
        scale: tuple[float, float],
        new_offset: tuple[float, float],
    ) -> tuple[float, float]:
        return (
            scale[0] * new_offset[0] + offset[0],
            scale[1] * new_offset[1] + offset[1],
        )

    @staticmethod
    def update_scale(
        scale: tuple[float, float], new_scale: tuple[float, float]
    ) -> tuple[float, float]:
        return scale[0] * new_scale[0], scale[1] * new_scale[1]

    @staticmethod
    def update_offsets_and_scales(context, new_offsets, new_scales) -> tuple:
        """
        x = x' * scale' + offset'
        x' = x'' * scale'' + offset''
        -> x = x'' * (scale' * scale'') + (scale' * offset'' + offset')
        """
        # scales and offsets are either both lists or both tuples
        offsets = context.get("offsets", (0, 0))
        scales = context.get("scales", (1, 1))
        if isinstance(offsets, tuple):
            if isinstance(new_offsets, list):
                updated_offsets = [
                    AugmentImage.update_offset(offsets, scales, new_offset)
                    for new_offset in new_offsets
                ]
                updated_scales = [
                    AugmentImage.update_scale(scales, new_scale)
                    for new_scale in new_scales
                ]
            else:
                if not len(offsets) == len(new_offsets):
                    raise ValueError("Cannot rescale lists when not same length")

                updated_offsets = AugmentImage.update_offset(
                    offsets, scales, new_offsets
                )
                updated_scales = AugmentImage.update_scale(scales, new_scales)
        else:
            if isinstance(new_offsets, list):
                if not len(offsets) == len(new_offsets):
                    raise ValueError("Cannot rescale lists when not same length")

                updated_offsets = [
                    AugmentImage.update_offset(offset, scale, new_offset)
                    for offset, scale, new_offset in zip(offsets, scales, new_offsets)
                ]
                updated_scales = [
                    AugmentImage.update_scale(scale, new_scale)
                    for scale, new_scale in zip(scales, new_scales)
                ]
            else:
                updated_offsets = [
                    AugmentImage.update_offset(offset, scale, new_offsets)
                    for offset, scale in zip(offsets, scales)
                ]
                updated_scales = [
                    AugmentImage.update_scale(scale, new_scales) for scale in scales
                ]
        return updated_offsets, updated_scales

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        # If the image is a batch, process each entry
        if len(image.shape) == 4:
            batch_size, h, w, _ = image.shape
            if batch_size == 0:
                # no images in top-down when no detections
                offsets, scales = (0, 0), (1, 1)
            else:
                transformed = [
                    self.transform(
                        image=img,
                        keypoints=[],
                        class_labels=[],
                        bboxes=[[0, 0, w, h]],
                        bbox_labels=["image"],
                    )
                    for img in image
                ]
                image = np.stack([t["image"] for t in transformed])
                output_bboxes = [t["bboxes"][0] for t in transformed]
                offsets, scales = self.get_offsets_and_scales(h, w, output_bboxes)
        else:
            h, w, _ = image.shape
            transformed = self.transform(
                image=image,
                keypoints=[],
                class_labels=[],
                bboxes=[[0, 0, w, h]],
                bbox_labels=["image"],
            )
            image = transformed["image"]
            output_bboxes = [transformed["bboxes"][0]]
            offsets, scales = self.get_offsets_and_scales(h, w, output_bboxes)
            offsets = offsets[0]
            scales = scales[0]

        offsets, scales = self.update_offsets_and_scales(context, offsets, scales)
        context["offsets"] = offsets
        context["scales"] = scales
        return image, context


class ToTensor(Preprocessor):
    """Transforms lists and numpy arrays into tensors"""

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        image = torch.tensor(image, dtype=torch.float)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        else:
            image = image.permute(2, 0, 1)
        return image, context


class ToBatch(Preprocessor):
    """TODO"""

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        return image.unsqueeze(0), context


class TorchCropDetections(Preprocessor):
    """TODO"""

    def __init__(self, cropped_image_size: int, bbox_format: str = "xywh") -> None:
        self.cropped_image_size = cropped_image_size
        self.bbox_format = bbox_format

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        """TODO: numpy implementation"""
        if "bboxes" not in context:
            raise ValueError(f"Must include bboxes to CropDetections, found {context}")

        images, offsets, scales = [], [], []
        for bbox in context["bboxes"]:
            cropped_image, offset, scale = _crop_and_pad_image_torch(
                image, bbox, "xywh", self.cropped_image_size
            )
            images.append(cropped_image)
            offsets.append(offset)
            scales.append(scale)

        context["offsets"] = np.array(offsets)
        context["scales"] = np.array(scales)

        # can have no bounding boxes if detector made no detections
        if len(images) == 0:
            images = np.zeros((0, *image.shape))
        else:
            images = np.stack(images, axis=0)

        return images, context
