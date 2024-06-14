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
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterable

import numpy as np
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.data.postprocessor import Postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import Preprocessor
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import ModelType, Runner
from deeplabcut.pose_estimation_pytorch.task import Task


class InferenceRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
    """Base class for inference runners

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        device: str = "cpu",
        snapshot_path: str | Path | None = None,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
    ):
        """
        Args:
            model: the model to run actions on
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            snapshot_path: if defined, the path of a snapshot from which to load pretrained weights
            preprocessor: the preprocessor to use on images before inference
            postprocessor: the postprocessor to use on images after inference
        """
        super().__init__(model=model, device=device, snapshot_path=snapshot_path)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        if self.snapshot_path is not None and self.snapshot_path != "":
            self.load_snapshot(self.snapshot_path, self.device, self.model)

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            the predictions for each of the 'batch_size' inputs
        """

    @torch.no_grad()
    def inference(
        self,
        images: Iterable[str | np.ndarray]
        | Iterable[tuple[str | np.ndarray, dict[str, Any]]],
    ) -> list[dict[str, np.ndarray]]:
        """Run model inference on the given dataset

        TODO: Add an option to also return head outputs (such as heatmaps)? Can be
         super useful for debugging

        Args:
            images: the images to run inference on, optionally with context

        Returns:
            a dict containing head predictions for each image
            [
                {
                    "bodypart": {"poses": np.array},
                    "unique_bodypart": "poses": np.array},
                }
            ]
        """
        self.model.to(self.device)
        self.model.eval()

        results = []
        for data in images:
            if isinstance(data, (str, np.ndarray)):
                input_image, context = data, {}
            else:
                input_image, context = data

            if self.preprocessor is not None:
                # TODO: input batch should also be able to be a dict[str, torch.Tensor]
                input_image, context = self.preprocessor(input_image, context)

            image_predictions = self.predict(input_image)
            if self.postprocessor is not None:
                # TODO: Should we return context?
                # TODO: typing update - the post-processor can remove a dict level
                image_predictions, _ = self.postprocessor(image_predictions, context)

            results.append(image_predictions)

        return results


class PoseInferenceRunner(InferenceRunner[PoseModel]):
    """Runner for pose estimation inference"""

    def __init__(self, model: PoseModel, **kwargs):
        super().__init__(model, **kwargs)

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": "poses": np.ndarray},
            ]
        """
        # TODO: iterates over batch one element at a time
        batch_size = 1
        batch_predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_inputs = batch_inputs.to(self.device)
            batch_outputs = self.model(batch_inputs)
            raw_predictions = self.model.get_predictions(batch_outputs)

            for b in range(batch_size):
                image_predictions = {}
                for head, head_outputs in raw_predictions.items():
                    image_predictions[head] = {}
                    for pred_name, pred in head_outputs.items():
                        image_predictions[head][pred_name] = pred[b].cpu().numpy()
                batch_predictions.append(image_predictions)

        return batch_predictions


class DetectorInferenceRunner(InferenceRunner[BaseDetector]):
    """Runner for object detection inference"""

    def __init__(self, model: BaseDetector, **kwargs):
        """
        Args:
            model: The detector to use for inference.
            **kwargs: Inference runner kwargs.
        """
        super().__init__(model, **kwargs)

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": "poses": np.ndarray},
            ]
        """
        # TODO: iterates over batch one element at a time
        batch_size = 1
        batch_predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_inputs = batch_inputs.to(self.device)
            _, raw_predictions = self.model(batch_inputs)
            for b, item in enumerate(raw_predictions):
                # take the top-k bounding boxes as individuals
                batch_predictions.append(
                    {
                        "detection": {
                            "bboxes": item["boxes"]
                            .cpu()
                            .numpy()
                            .reshape(-1, 4),
                            "scores": item["scores"]
                            .cpu()
                            .numpy()
                            .reshape(-1),
                        }
                    }
                )

        return batch_predictions


def build_inference_runner(
    task: Task,
    model: nn.Module,
    device: str,
    snapshot_path: str | Path,
    preprocessor: Preprocessor | None = None,
    postprocessor: Postprocessor | None = None,
) -> InferenceRunner:
    """
    Build a runner object according to a pytorch configuration file

    Args:
        task: the inference task to run
        model: the model to run
        device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
        snapshot_path: the snapshot from which to load the weights
        preprocessor: the preprocessor to use on images before inference
        postprocessor: the postprocessor to use on images after inference

    Returns:
        the inference runner
    """
    kwargs = dict(
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    if task == Task.DETECT:
        return DetectorInferenceRunner(**kwargs)

    return PoseInferenceRunner(**kwargs)
