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
from collections import defaultdict
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
        batch_size: int = 1,
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
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; is {batch_size}")

        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        if self.snapshot_path is not None and self.snapshot_path != "":
            self.load_snapshot(self.snapshot_path, self.device, self.model)

        self._batch: torch.Tensor | None = None
        self._contexts: list[dict] = []
        self._image_batch_sizes: list[int] = []
        self._predictions: list = []

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
            self._prepare_inputs(data)
            self._process_full_batches()
            results += self._extract_results()

        # Process the last batch even if not full
        if self._inputs_waiting_for_processing():
            self._process_batch()
            results += self._extract_results()

        return results

    def _prepare_inputs(
        self, data: str | np.ndarray | tuple[str | np.ndarray, dict],
    ) -> None:
        """
        Prepares inputs for an image and adds them to the data ready to be processed
        """
        if isinstance(data, (str, np.ndarray)):
            inputs, context = data, {}
        else:
            inputs, context = data

        if self.preprocessor is not None:
            inputs, context = self.preprocessor(inputs, context)
        else:
            inputs = torch.as_tensor(inputs)

        self._contexts.append(context)
        self._image_batch_sizes.append(len(inputs))

        # skip when there are no inputs for an image
        if len(inputs) == 0:
            return

        if self._batch is None:
            self._batch = inputs
        else:
            self._batch = torch.cat([self._batch, inputs], dim=0)

    def _process_full_batches(self) -> None:
        """Processes prepared inputs in batches of the desired batch size."""
        while self._batch is not None and len(self._batch) >= self.batch_size:
            self._process_batch()

    def _extract_results(self) -> list:
        """Obtains results that were obtained from processing a batch."""
        results = []
        while (
            len(self._image_batch_sizes) > 0
            and len(self._predictions) >= self._image_batch_sizes[0]
        ):
            num_predictions = self._image_batch_sizes[0]
            image_predictions = self._predictions[:num_predictions]
            context = self._contexts[0]

            if self.postprocessor is not None:
                # TODO: Should we return context?
                # TODO: typing update - the post-processor can remove a dict level
                image_predictions, _ = self.postprocessor(image_predictions, context)

            self._contexts = self._contexts[1:]
            self._image_batch_sizes = self._image_batch_sizes[1:]
            self._predictions = self._predictions[num_predictions:]
            results.append(image_predictions)

        return results

    def _process_batch(self) -> None:
        """
        Processes a batch. There must be inputs waiting to be processed before this is
        called, otherwise this method will raise an error.
        """
        batch = self._batch[:self.batch_size]
        self._predictions += self.predict(batch)

        # remove processed inputs from batch
        if len(self._batch) <= self.batch_size:
            self._batch = None
        else:
            self._batch = self._batch[self.batch_size:]

    def _inputs_waiting_for_processing(self) -> bool:
        """Returns: Whether there are inputs which have not yet been processed"""
        return self._batch is not None and len(self._batch) > 0


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
        outputs = self.model(inputs.to(self.device))
        raw_predictions = self.model.get_predictions(outputs)
        predictions = [
            {
                head: {
                    pred_name: pred[b].cpu().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
            for b in range(len(inputs))
        ]
        return predictions


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
        _, raw_predictions = self.model(inputs.to(self.device))
        predictions = [
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
            for item in raw_predictions
        ]
        return predictions


def build_inference_runner(
    task: Task,
    model: nn.Module,
    device: str,
    snapshot_path: str | Path,
    batch_size: int = 1,
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
        batch_size: the batch size to use to run inference
        preprocessor: the preprocessor to use on images before inference
        postprocessor: the postprocessor to use on images after inference

    Returns:
        the inference runner
    """
    kwargs = dict(
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    if task == Task.DETECT:
        return DetectorInferenceRunner(**kwargs)

    return PoseInferenceRunner(**kwargs)
