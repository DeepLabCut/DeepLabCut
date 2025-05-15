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
from typing import Any, TypeVar, Callable

import albumentations as A
import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.data.image import load_image, top_down_crop
from deeplabcut.pose_estimation_pytorch.data.utils import bbox_from_keypoints


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
    color_mode: str,
    transform: A.BaseCompose,
    top_down_crop_size: tuple[int, int],
    top_down_crop_margin: int = 0,
    top_down_crop_with_context: bool = True,
) -> Preprocessor:
    """Creates a preprocessor for top-down pose estimation

    Creates a preprocessor that loads an image, crops all bounding boxes given as a
    context (through a "bboxes" key), runs some transforms on each cropped image (such
    as normalization), creates a tensor from the numpy array (going from
    (num_ind, h, w, 3) to (num_ind, 3, h, w)).

    Args:
        color_mode: whether to load the image as an RGB or BGR
        transform: the transform to apply to the image
        top_down_crop_size: the (width, height) to resize cropped bboxes to
        top_down_crop_margin: the margin to add around detected bboxes for the crop
        top_down_crop_with_context: whether to keep context when applying the top-down crop

    Returns:
        A default top-down Preprocessor
    """
    return ComposePreprocessor(
        components=[
            LoadImage(color_mode),
            TopDownCrop(
                output_size=top_down_crop_size,
                margin=top_down_crop_margin,
                with_context=top_down_crop_with_context,
            ),
            AugmentImage(transform),
            ToTensor(),
        ]
    )


def build_conditional_top_down_preprocessor(
    color_mode: str,
    transform: A.BaseCompose,
    bbox_margin: int,
    top_down_crop_size: tuple[int, int],
    top_down_crop_margin: int = 0,
    top_down_crop_with_context: bool = False,
) -> Preprocessor:
    """Creates a preprocessor for conditional top-down pose estimation

    Creates a preprocessor that loads an image, computes bounding boxes from conditional
    keypoints (given as a context (through a "cond_kpts" key), crops all bounding boxes,
    runs some transforms on each cropped image (such as normalization), creates a tensor
    from the numpy array (going from (num_ind, h, w, 3) to (num_ind, 3, h, w)).

    Args:
        color_mode: whether to load the image as an RGB or BGR
        transform: the transform to apply to the image
        bbox_margin: The margin to add around keypoints when generating bounding boxes
            from conditional keypoints.
        top_down_crop_size: the (width, height) to resize cropped bboxes to
        top_down_crop_margin: the margin to add around detected bboxes for the crop
        top_down_crop_with_context: whether to keep context when applying the top-down crop

    Returns:
        A default conditional top-down Preprocessor
    """
    return ComposePreprocessor(
        components=[
            LoadImage(color_mode),
            FilterLowConfidencePoses(),
            ComputeBoundingBoxesFromCondKeypoints(bbox_margin=bbox_margin),
            FilterInvalidBoundingBoxes(),
            TopDownCrop(
                output_size=top_down_crop_size,
                margin=top_down_crop_margin,
                with_context=top_down_crop_with_context,
            ),
            AugmentImage(transform),
            ConditionalKeypointsToModelInputs(),
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

    def __init__(self, color_mode: str = "RGB") -> None:
        self.color_mode = color_mode

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode=self.color_mode)

        h, w = image.shape[:2]
        context["image_size"] = w, h
        return image, context


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
    """Adds a batch dimension to the image tensor.

    This preprocessor is used to convert a single image tensor into a batched format
    by unsqueezing along the 0th dimension. This is typically required before passing
    the image to models that expect batched input (i.e., shape `[B, C, H, W]`).
    """

    def __call__(self, image: Image, context: Context) -> tuple[np.ndarray, Context]:
        return image.unsqueeze(0), context


class FilterLowConfidencePoses(Preprocessor):
    """
    Filters out poses with low confidence scores.
    By default, the confidence associated to the pose is the max confidence value.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.05,
        aggregate_func: Callable[[np.ndarray], float] = lambda arr: np.max(arr, axis=1),
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.aggregate_func = aggregate_func

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        if "cond_kpts" not in context:
            raise ValueError(f"Must include cond_kpts, found {context}")

        keypoints = context["cond_kpts"]
        mask = self.aggregate_func(keypoints[:, :, 2]) >= self.confidence_threshold
        context["cond_kpts"] = keypoints[mask]

        return image, context


class FilterInvalidBoundingBoxes(Preprocessor):
    """Filters out poses and bounding boxes that are invalid (e.g., area too small)."""

    def __init__(self, min_area: int = 1) -> None:
        self.min_area = min_area

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        bboxes = context.get("bboxes", [])
        keypoints = context.get("cond_kpts", [])

        valid_bboxes = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            _, _, w, h = bbox
            if w * h >= self.min_area:
                valid_bboxes.append(bbox)
                valid_indices.append(i)

        context["bboxes"] = valid_bboxes
        context["cond_kpts"] = keypoints[valid_indices]

        return image, context


class TopDownCrop(Preprocessor):
    """Crops bounding boxes out of images for top-down pose estimation

    Args:
        output_size: The (width, height) of crops to output
        margin: The margin to add around detected bounding boxes before cropping
        with_context: Whether to keep context in the top-down crop
    """

    def __init__(
        self,
        output_size: int | tuple[int, int],
        margin: int = 0,
        with_context: bool = True,
    ) -> None:
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = output_size
        self.margin = margin
        self.with_context = with_context

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        """TODO: numpy implementation"""
        if "bboxes" not in context:
            raise ValueError(f"Must include bboxes to CropDetections, found {context}")

        images, offsets, scales = [], [], []
        for bbox in context["bboxes"]:
            crop, offset, scale = top_down_crop(
                image,
                bbox,
                self.output_size,
                margin=self.margin,
                crop_with_context=self.with_context,
            )
            images.append(crop)
            offsets.append(offset)
            scales.append(scale)

        context["offsets"] = np.array(offsets)
        context["scales"] = np.array(scales)

        # can have no bounding boxes if detector made no detections
        if len(images) == 0:
            images = np.zeros((0, *image.shape))
        else:
            images = np.stack(images, axis=0)

        context["top_down_crop_size"] = self.output_size
        return images, context


class ComputeBoundingBoxesFromCondKeypoints(Preprocessor):
    """Generates bounding boxes from predicted keypoints

    Args:
        cond_kpt_key: The key under which cond. keypoints are stored in the context.
        bbox_margin: The margin to add around keypoints when generating bounding boxes.
    """

    def __init__(self, cond_kpt_key: str = "cond_kpts", bbox_margin: int = 0) -> None:
        self.cond_kpt_key = cond_kpt_key
        self.bbox_margin = bbox_margin

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        """TODO: numpy implementation"""
        if "cond_kpts" not in context:
            raise ValueError(
                f"Must include cond kpts to ComputeBBoxes, found {context}"
            )

        h, w = image.shape[:2]
        context["bboxes"] = [
            bbox_from_keypoints(cond_kpts, h, w, self.bbox_margin)
            for cond_kpts in context[self.cond_kpt_key]
        ]
        return image, context


class ConditionalKeypointsToModelInputs(Preprocessor):

    def __init__(self, cond_kpt_key: str = "cond_kpts") -> None:
        self.cond_kpt_key = cond_kpt_key

    def __call__(
        self, image: np.ndarray, context: Context
    ) -> tuple[np.ndarray, Context]:
        cond_keypoints = context[self.cond_kpt_key]
        if len(cond_keypoints) == 0:
            return image, context

        rescaled = cond_keypoints.copy()
        rescaled[..., :2] = (
            rescaled[..., :2] - np.array(context["offsets"])[:, None]
        ) / np.array(context["scales"])[:, None]
        context["model_kwargs"] = {"cond_kpts": np.expand_dims(rescaled, axis=1)}
        return image, context
