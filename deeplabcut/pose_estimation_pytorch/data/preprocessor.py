"""Helpers to run preprocess data before running inference"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Any

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
            context: the context for this image or batch of images (such as )

        Returns:
            the pre-processed image (or batch of images) and their context
        """
        pass


def build_bottom_up_preprocessor(
    color_mode: str,
    transform: A.BaseCompose,
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
    color_mode: str,
    transform: A.BaseCompose,
    cropped_image_size: tuple[int, int],
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
    """TODO"""

    def __init__(self, transform: A.BaseCompose) -> None:
        self.transform = transform

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        # If the image is a batch, process each entry
        if len(image.shape) == 4:
            transformed = [
                self.transform(
                    image=img, keypoints=[], class_labels=[], bboxes=[], bbox_labels=[]
                )["image"]
                for img in image
            ]
            image = np.stack(transformed)
        else:
            image = self.transform(
                image=image, keypoints=[], class_labels=[], bboxes=[], bbox_labels=[]
            )["image"]

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

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
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

        context["offsets"] = offsets
        context["scales"] = scales
        return np.stack(images, axis=0), context
