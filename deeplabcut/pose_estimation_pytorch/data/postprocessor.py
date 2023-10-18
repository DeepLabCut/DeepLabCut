"""Post-process predictions made by models"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from deeplabcut.pose_estimation_pytorch.data.preprocessor import Context


class Postprocessor(ABC):
    """A post-processor can be called on the output of a model
    TODO: Documentation
    """

    @abstractmethod
    def __call__(self, predictions: Any, context: Context) -> Any:
        """
        Post-processes the outputs of a model into a single prediction.

        Args:
            predictions: the predictions made by the model on a single image
            context: the context returned by the pre-processor with the image

        Returns:
            a single post-processed prediction
        """
        pass


def build_bottom_up_postprocessor(with_unique_bodyparts: bool) -> ComposePostprocessor:
    """Creates a postprocessor for bottom-up pose estimation (or object detection)

    Args:
        with_unique_bodyparts: whether the model outputs unique bodyparts

    Returns:
        A default bottom-up Postprocessor
    """
    keys_to_concatenate = {"bodyparts": ("bodypart", "poses")}
    if with_unique_bodyparts:
        keys_to_concatenate["unique_bodyparts"] = ("unique_bodypart", "poses")
    return ComposePostprocessor(
        components=[
            ConcatenateOutputs(keys_to_concatenate=keys_to_concatenate),
        ]
    )


def build_top_down_postprocessor(with_unique_bodyparts: bool) -> Postprocessor:
    """Creates a postprocessor for top-down pose estimation

    Args:
        with_unique_bodyparts: whether the model outputs unique bodyparts

    Returns:
        A default top-down Postprocessor
    """
    keys_to_concatenate = {"bodyparts": ("bodypart", "poses")}
    keys_to_rescale = ["bodyparts"]
    if with_unique_bodyparts:
        keys_to_concatenate["unique_bodyparts"] = ("unique_bodypart", "poses")
        keys_to_rescale.append("unique_bodyparts")
    return ComposePostprocessor(
        components=[
            ConcatenateOutputs(keys_to_concatenate=keys_to_concatenate),
            RescaleAndOffset(keys_to_rescale=keys_to_rescale),
        ]
    )


def build_detector_postprocessor() -> Postprocessor:
    """Creates a postprocessor for top-down pose estimation

    Returns:
        A default top-down Postprocessor
    """
    return ComposePostprocessor(
        components=[
            ConcatenateOutputs(keys_to_concatenate={"bboxes": ("detection", "bboxes")}),
            BboxToCoco(bounding_box_keys=["bboxes"]),
        ]
    )


class ComposePostprocessor(Postprocessor):
    """
    Class to preprocess an image and turn it into a batch of
    inputs before running inference
    """

    def __init__(self, components: list[Postprocessor]) -> None:
        self.components = components

    def __call__(self, predictions: Any, context: Context) -> tuple[Any, Context]:
        for postprocessor in self.components:
            predictions, context = postprocessor(predictions, context)
        return predictions, context


class ConcatenateOutputs(Postprocessor):
    """Checks that there is a single prediction for the image and returns it"""

    def __init__(self, keys_to_concatenate: dict[str, tuple[str, str]]):
        self.keys_to_concatenate = keys_to_concatenate

    def __call__(
        self, predictions: Any, context: Context
    ) -> tuple[dict[str, np.ndarray], Context]:
        if len(predictions) == 0:
            raise ValueError("Cannot concatenate outputs: predictions has length 0")

        outputs = {}
        for output_name, head_key in self.keys_to_concatenate.items():
            head_name, val_name = head_key
            outputs[output_name] = np.concatenate(
                [p[head_name][val_name] for p in predictions]
            )

        return outputs, context


class RescaleAndOffset(Postprocessor):
    """Rescales and offsets images back to their position in the original image"""

    def __init__(self, keys_to_rescale: list[str]) -> None:
        super().__init__()
        self.keys_to_rescale = keys_to_rescale

    def __call__(
        self,
        predictions: dict[str, np.ndarray],
        context: Context,
    ) -> tuple[dict[str, np.ndarray], Context]:
        if "scales" not in context or "offsets" not in context:
            raise ValueError(
                "RescalePostprocessor needs 'scales' and 'offsets' in the context, "
                f"found {context}"
            )

        updated_predictions = {}
        scales, offsets = np.array(context["scales"]), np.array(context["offsets"])
        for name, outputs in predictions.items():
            if name in self.keys_to_rescale:
                if not len(outputs) == len(scales) == len(offsets):
                    raise ValueError(
                        "There must be as many 'scales' and 'offsets' as outputs, found "
                        f"{len(outputs)}, {len(scales)}, {len(offsets)}"
                    )

                rescaled = []
                for output, scale, offset in zip(outputs, scales, offsets):
                    output_rescaled = output.copy()
                    output_rescaled[:, 0] = output[:, 0] * scale[0] + offset[0]
                    output_rescaled[:, 1] = output[:, 1] * scale[1] + offset[1]
                    rescaled.append(output_rescaled)
                updated_predictions[name] = np.stack(rescaled)
            else:
                updated_predictions[name] = outputs.copy()

        return updated_predictions, context


class BboxToCoco(Postprocessor):
    """Transforms bounding boxes from xyxy to COCO format (xywh)"""

    def __init__(self, bounding_box_keys: list[str]) -> None:
        super().__init__()
        self.bounding_box_keys = bounding_box_keys

    def __call__(
        self,
        predictions: dict[str, np.ndarray],
        context: Context,
    ) -> tuple[dict[str, np.ndarray], Context]:
        for bbox_key in self.bounding_box_keys:
            predictions[bbox_key][:, 2] -= predictions[bbox_key][:, 0]
            predictions[bbox_key][:, 3] -= predictions[bbox_key][:, 1]

        return predictions, context
