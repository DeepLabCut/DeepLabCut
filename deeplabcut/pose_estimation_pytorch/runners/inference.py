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
import threading
from queue import Queue, Empty
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

import deeplabcut.pose_estimation_pytorch.post_processing.nms as nms
import deeplabcut.pose_estimation_pytorch.runners.ctd as ctd
import deeplabcut.pose_estimation_pytorch.runners.shelving as shelving
from deeplabcut.core.inferenceutils import calc_object_keypoint_similarity
from deeplabcut.pose_estimation_pytorch.data.postprocessor import Postprocessor
from deeplabcut.pose_estimation_pytorch.data.preprocessor import LoadImage, Preprocessor
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import ModelType, Runner
from deeplabcut.pose_estimation_pytorch.runners.dynamic_cropping import (
    DynamicCropper,
    TopDownDynamicCropper,
)
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
        load_weights_only: bool | None = None,
        async_mode: bool = True,
        num_prefetch_batches: int = 2,
        timeout: float = 30.0,
    ):
        """
        Args:
            model: The model to run actions on
            device: The device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            snapshot_path: If defined, the path of a snapshot from which to load
                pretrained weights
            preprocessor: The preprocessor to use on images before inference
            postprocessor: The postprocessor to use on images after inference
            load_weights_only: Value for the torch.load() `weights_only` parameter.
                If False, the python pickle module is used implicitly, which is known to
                    be insecure. Only set to False if you're loading data that you trust
                    (e.g. snapshots that you created). For more information, see:
                        https://pytorch.org/docs/stable/generated/torch.load.html
                If None, the default value is used:
                    `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`
            async_mode: Whether to use async inference with pipeline parallelism
            num_prefetch_batches: Number of batches to prefetch in async mode
            timeout: Timeout for queue operations in async mode
        """
        super().__init__(model=model, device=device, snapshot_path=snapshot_path)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer; is {batch_size}")

        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.async_mode = async_mode
        self.num_prefetch_batches = num_prefetch_batches
        self.timeout = timeout

        if self.snapshot_path is not None and self.snapshot_path != "":
            self.load_snapshot(
                self.snapshot_path,
                self.device,
                self.model,
                weights_only=load_weights_only,
            )

        self.model.to(self.device)
        self.model.eval()

        self._batch: torch.Tensor | None = None
        self._model_kwargs: dict[str, np.ndarray | torch.Tensor] = {}

        self._contexts: list[dict] = []
        self._image_batch_sizes: list[int] = []
        self._predictions: list = []

        # Async-specific attributes
        if self.async_mode:
            self._input_queue = Queue(maxsize=num_prefetch_batches)
            self._preprocessing_thread = None
            self._stop_event = threading.Event()
            self._exception = None

    @abstractmethod
    def predict(
        self, inputs: torch.Tensor, **kwargs
    ) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            the predictions for each of the 'batch_size' inputs
        """

    @torch.inference_mode()
    def inference(
        self,
        images: (
            Iterable[str | Path | np.ndarray]
            | Iterable[tuple[str | Path | np.ndarray, dict[str, Any]]]
        ),
        shelf_writer: shelving.ShelfWriter | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Run model inference on the given dataset

        TODO: Add an option to also return head outputs (such as heatmaps)? Can be
         super useful for debugging

        Args:
            images: the images to run inference on, optionally with context
            shelf_writer: by default, data are saved in a list and returned at the end
                of inference. Passing a shelf manager writes data to disk on-the-fly
                using a "shelf" (a pickle-based, persistent, database-like object by
                default, resulting in constant memory footprint). The returned list is
                then empty.

        Returns:
            a dict containing head predictions for each image
            [
                {
                    "bodypart": {"poses": np.array},
                    "unique_bodypart": {"poses": np.array},
                }
            ]
        """
        if self.async_mode:
            return self._async_inference(images, shelf_writer)
        else:
            return self._sequential_inference(images, shelf_writer)

    def _sequential_inference(
        self,
        images: (
            Iterable[str | Path | np.ndarray]
            | Iterable[tuple[str | Path | np.ndarray, dict[str, Any]]]
        ),
        shelf_writer: shelving.ShelfWriter | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Original sequential inference implementation"""
        results = []
        for data in images:
            self._prepare_inputs(data)
            self._process_full_batches()
            results += self._extract_results(shelf_writer)

        # Process the last batch even if not full
        if self._inputs_waiting_for_processing():
            self._process_batch()
            results += self._extract_results(shelf_writer)

        return results

    def _async_inference(
        self,
        images: (
            Iterable[str | Path | np.ndarray]
            | Iterable[tuple[str | Path | np.ndarray, dict[str, Any]]]
        ),
        shelf_writer: shelving.ShelfWriter | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Async inference with pipeline parallelism"""
        # Reset state
        self._stop_event.clear()
        self._exception = None
        self._batch = None
        self._model_kwargs = {}
        self._contexts = []
        self._image_batch_sizes = []
        self._predictions = []

        # Start preprocessing thread
        self._preprocessing_thread = threading.Thread(
            target=self._preprocessing_worker, args=(images,)
        )
        self._preprocessing_thread.start()

        results = []

        try:
            while True:
                # Get next batch from queue
                try:
                    item = self._input_queue.get(timeout=self.timeout)
                except Empty:
                    # Check if preprocessing thread is still alive
                    if self._preprocessing_thread.is_alive():
                        continue
                    else:
                        break

                if item is None:
                    # Preprocessing is done
                    break

                batch, model_kwargs = item

                # Run model inference
                predictions = self.predict(batch, **model_kwargs)
                self._predictions.extend(predictions)

                # Extract and return results
                batch_results = self._extract_results(shelf_writer)
                results.extend(batch_results)

        except Exception as e:
            self._stop_event.set()
            raise e
        finally:
            # Wait for preprocessing thread to finish
            if self._preprocessing_thread is not None:
                self._preprocessing_thread.join(timeout=self.timeout)

            # Check for exceptions in preprocessing thread
            if self._exception is not None:
                raise self._exception

        return results

    def _prepare_inputs(
        self,
        data: str | Path | np.ndarray | tuple[str | Path | np.ndarray, dict],
    ) -> None:
        """
        Prepares inputs for an image and adds them to the data ready to be processed
        """
        if isinstance(data, (str, Path, np.ndarray)):
            inputs, context = data, {}
        else:
            inputs, context = data

        if self.preprocessor is not None:
            inputs, context = self.preprocessor(inputs, context)
        else:
            inputs = torch.as_tensor(inputs)

        # add new model_kwargs from the inputs
        model_kwargs = context.pop("model_kwargs", {})
        for k, v in model_kwargs.items():
            curr_v = self._model_kwargs.get(k)
            if curr_v is None or len(curr_v) == 0:
                curr_v = v
            elif len(v) == 0:
                continue
            elif isinstance(curr_v, np.ndarray):
                curr_v = np.concatenate([curr_v, v], axis=0)
            elif isinstance(curr_v, torch.Tensor):
                curr_v = torch.cat([curr_v, v], dim=0)
            else:
                raise ValueError(
                    f"model_kwargs {k} must be a numpy array or torch tensor - "
                    f"found '{type(v)}'."
                )
            self._model_kwargs[k] = curr_v

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

    def _extract_results(self, shelf_writer: shelving.ShelfWriter) -> list:
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

            if shelf_writer is not None:
                shelf_writer.add_prediction(
                    bodyparts=image_predictions["bodyparts"],
                    unique_bodyparts=image_predictions.get("unique_bodyparts"),
                    identity_scores=image_predictions.get("identity_scores"),
                    features=image_predictions.get("features"),
                )
            else:
                results.append(image_predictions)

            self._contexts = self._contexts[1:]
            self._image_batch_sizes = self._image_batch_sizes[1:]
            self._predictions = self._predictions[num_predictions:]

        return results

    def _process_batch(self) -> None:
        """
        Processes a batch. There must be inputs waiting to be processed before this is
        called, otherwise this method will raise an error.
        """
        batch = self._batch[: self.batch_size]
        model_kwargs = {
            mk: v[: self.batch_size] for mk, v in self._model_kwargs.items()
        }

        self._predictions += self.predict(batch, **model_kwargs)

        # remove processed inputs from batch
        if len(self._batch) <= self.batch_size:
            self._batch = None
            self._model_kwargs = {}
        else:
            self._batch = self._batch[self.batch_size :]
            self._model_kwargs = {
                mk: v[self.batch_size :] for mk, v in self._model_kwargs.items()
            }

    def _inputs_waiting_for_processing(self) -> bool:
        """Returns: Whether there are inputs which have not yet been processed"""
        return self._batch is not None and len(self._batch) > 0

    def _preprocessing_worker(self, images: Iterable) -> None:
        """Background worker that prepares inputs and puts them in the input queue"""
        try:
            for data in images:
                if self._stop_event.is_set():
                    break

                # Prepare inputs using the parent class method
                self._prepare_inputs(data)

                # Process full batches and put them in the queue
                while self._batch is not None and len(self._batch) >= self.batch_size:
                    batch = self._batch[: self.batch_size]
                    model_kwargs = {
                        mk: v[: self.batch_size] for mk, v in self._model_kwargs.items()
                    }

                    # Put the batch in the queue for processing
                    self._input_queue.put((batch, model_kwargs), timeout=self.timeout)

                    # Remove processed inputs from batch
                    if len(self._batch) <= self.batch_size:
                        self._batch = None
                        self._model_kwargs = {}
                    else:
                        self._batch = self._batch[self.batch_size :]
                        self._model_kwargs = {
                            mk: v[self.batch_size :]
                            for mk, v in self._model_kwargs.items()
                        }

            # Process any remaining inputs
            if self._batch is not None and len(self._batch) > 0:
                batch = self._batch
                model_kwargs = self._model_kwargs
                self._input_queue.put((batch, model_kwargs), timeout=self.timeout)

        except Exception as e:
            self._exception = e
            self._stop_event.set()
        finally:
            # Signal that preprocessing is done
            self._input_queue.put(None, timeout=self.timeout)

    def __del__(self):
        """Cleanup method to ensure threads are stopped"""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if (
            hasattr(self, "_preprocessing_thread")
            and self._preprocessing_thread is not None
        ):
            self._preprocessing_thread.join(timeout=1.0)


class PoseInferenceRunner(InferenceRunner[PoseModel]):
    """Runner for pose estimation inference"""

    def __init__(
        self,
        model: PoseModel,
        dynamic: DynamicCropper | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.dynamic = dynamic
        if dynamic is not None and self.batch_size != 1:
            raise ValueError(
                "Dynamic cropping can only be used with batch size 1. Please set "
                "your batch size to 1."
            )

    def predict(
        self, inputs: torch.Tensor, **kwargs
    ) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": {"poses": np.ndarray},
                }
            ]
        """
        batch_size = len(inputs)
        if self.dynamic is not None:
            # dynamic cropping can use patches
            inputs = self.dynamic.crop(inputs)
        if self.device and "cuda" in str(self.device):
            with torch.autocast(device_type=str(self.device)):
                outputs = self.model(inputs.to(self.device), **kwargs)
                raw_predictions = self.model.get_predictions(outputs)
        else:
            outputs = self.model(inputs.to(self.device), **kwargs)
            raw_predictions = self.model.get_predictions(outputs)

        if self.dynamic is not None:
            raw_predictions["bodypart"]["poses"] = self.dynamic.update(
                raw_predictions["bodypart"]["poses"]
            )

        predictions = [
            {
                head: {
                    pred_name: pred[b].cpu().numpy()
                    for pred_name, pred in head_outputs.items()
                }
                for head, head_outputs in raw_predictions.items()
            }
            for b in range(batch_size)
        ]
        return predictions


class CTDInferenceRunner(PoseInferenceRunner):
    """Runner for pose estimation inference

    Args:
        model: The CTD model to run inference with.
        bu_runner: A runner for the BU model to run inference with. If no BU runner is
            given, conditions must be given in the context for the data. Otherwise an
            error will be raised during inference.
        tracking: Whether to track using the CTD model. If
    """

    def __init__(
        self,
        model: PoseModel,
        bu_runner: PoseInferenceRunner | None = None,
        ctd_tracking: bool | ctd.CTDTrackingConfig = False,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.bu_runner = bu_runner
        if bu_runner is not None:
            self.bu_runner.model.eval()

        self.tracking = None
        if isinstance(ctd_tracking, ctd.CTDTrackingConfig):
            self.tracking = ctd_tracking
        elif ctd_tracking:  # generate default config
            self.tracking = ctd.CTDTrackingConfig()

        if self.tracking and self.batch_size != 1:
            print("CTD tracking can only be used with batch size 1. Updating it.")
            self.batch_size = 1

        self._image_loader = LoadImage()

        # Stored poses and IDX -> ID map for CTD tracking
        self._bu_age = -1
        self._missing_idvs = False
        self._prev_pose = None
        self._idx_to_id = None
        self._ctd_track_ages = None  # the age of each CTD tracklet

    @torch.inference_mode()
    def inference(
        self,
        images: (
            Iterable[str | Path | np.ndarray]
            | Iterable[tuple[str | Path | np.ndarray, dict[str, Any]]]
        ),
        shelf_writer: shelving.ShelfWriter | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Run CTD model inference on the given dataset

        Args:
            images: the images to run inference on, optionally with context
            shelf_writer: by default, data are saved in a list and returned at the end
                of inference. Passing a shelf manager writes data to disk on-the-fly
                using a "shelf" (a pickle-based, persistent, database-like object by
                default, resulting in constant memory footprint). The returned list is
                then empty.

        Returns:
            a dict containing head predictions for each image
            [
                {
                    "bodypart": {"poses": np.array},
                    "unique_bodypart": {"poses": np.array},
                }
            ]
        """
        if self.tracking:
            return self._ctd_tracking_inference(images, shelf_writer)

        results = []
        for data in images:
            data = self.add_conditions(data)
            self._prepare_inputs(data)
            self._process_full_batches()
            results += self._extract_results(shelf_writer)

        # Process the last batch even if not full
        if self._inputs_waiting_for_processing():
            self._process_batch()
            results += self._extract_results(shelf_writer)

        return results

    def predict(
        self, inputs: torch.Tensor, **kwargs
    ) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": {"poses": np.ndarray},
                }
            ]
        """
        if self.device and "cuda" in str(self.device):
            with torch.autocast(device_type=str(self.device)):
                outputs = self.model(inputs.to(self.device), **kwargs)
        else:
            outputs = self.model(inputs.to(self.device), **kwargs)
        raw_predictions = self.model.get_predictions(outputs)
        predictions = [
            {
                "detection": {
                    "bboxes": item["boxes"].cpu().numpy().reshape(-1, 4),
                    "bbox_scores": item["scores"].cpu().numpy().reshape(-1),
                }
            }
            for item in raw_predictions
        ]

        return predictions

    def add_conditions(
        self,
        data: str | Path | np.ndarray | tuple[str | Path | np.ndarray, dict],
    ) -> tuple[np.ndarray, dict]:
        if isinstance(data, (str, Path, np.ndarray)):
            inputs, context = data, {}
        else:
            inputs, context = data

        # Load the image once - then given as a numpy array to CTD
        image, _ = self._image_loader(inputs, context)

        # If the conditional keypoints are in the context, return the context
        if "cond_kpts" in context:
            return image, context

        # Run the pre-processor
        if self.bu_runner.preprocessor is not None:
            inputs, context = self.bu_runner.preprocessor(image, context)
        else:
            inputs = torch.as_tensor(image)

        # Get and post-process the predictions
        predictions = self.bu_runner.predict(inputs)
        if self.bu_runner.postprocessor is not None:
            predictions, context = self.bu_runner.postprocessor(predictions, context)

        # Extract the conditions
        conds = predictions["bodyparts"][..., :3]
        pred_mask = ~np.all(np.any(conds <= 0 | np.isnan(conds), axis=2), axis=1)
        if np.sum(pred_mask) > 0:
            conds = conds[pred_mask]
        else:
            conds = np.zeros((0, conds.shape[1], 3))

        return image, {"cond_kpts": conds}

    def _ctd_tracking_inference(
        self,
        images: (
            Iterable[str | Path | np.ndarray]
            | Iterable[tuple[str | Path | np.ndarray, dict[str, Any]]]
        ),
        shelf_writer: shelving.ShelfWriter | None = None,
    ) -> list[dict[str, np.ndarray]]:
        results = []
        for data in images:
            inputs, context = self._prepare_ctd_inputs(data)
            model_kwargs = context.pop("model_kwargs", {})
            predictions = self.predict(inputs, **model_kwargs)
            if self.postprocessor is not None:
                # Pop the "cond_kpts" from the context so there's no re-scoring
                # This is required when tracking with CTD, otherwise scores go to 0
                if self._prev_pose is not None:
                    context.pop("cond_kpts")

                predictions, _ = self.postprocessor(predictions, context)

            # Set the predictions as context for the next frame
            self._ctd_tracking_postprocess(predictions, context["image_size"])

            if shelf_writer is not None:
                shelf_writer.add_prediction(
                    bodyparts=predictions["bodyparts"],
                    unique_bodyparts=predictions.get("unique_bodyparts"),
                    identity_scores=predictions.get("identity_scores"),
                    features=predictions.get("features"),
                )
            else:
                results.append(predictions)

        return results

    def _prepare_ctd_inputs(self, data) -> tuple[torch.Tensor, dict[str, Any]]:
        # If there's no valid poses, use the BU model to get conditions
        self._bu_age += 1
        if (
            self._prev_pose is None
            or (
                self._missing_idvs
                and self.tracking.bu_on_lost_idv
                and self._bu_age >= self.tracking.bu_max_frequency
            )
            or (
                self.tracking.bu_min_frequency is not None
                and self._bu_age >= self.tracking.bu_min_frequency
            )
        ):
            self._bu_age = 0
            inputs, context = self.add_conditions(data)

            if self._prev_pose is not None:
                context["cond_kpts"] = self._merge_conditions(context["cond_kpts"])

        else:
            if isinstance(data, (str, Path, np.ndarray)):
                inputs, context = data, {}
            else:
                inputs, context = data

            context["cond_kpts"] = self._prev_pose

        if self.preprocessor is None:
            return torch.as_tensor(inputs), context

        inputs, context = self.preprocessor(inputs, context)
        return inputs, context

    def _ctd_tracking_postprocess(
        self,
        predictions: dict[str, np.ndarray],
        image_size: tuple[int, int],
    ) -> None:
        """Post-processes predictions. In-place changes to the predictions dict."""
        # reorder the previous poses so the indices match the track IDs
        if self._idx_to_id is not None:
            predictions["bodyparts"] = predictions["bodyparts"][self._idx_to_id]

        # mask all keypoints below the CTD tracking threshold
        prev_pose = predictions["bodyparts"][..., :3].copy()
        prev_pose[prev_pose[..., 2] <= self.tracking.threshold_ctd] = np.nan

        # mask all keypoints outside the image
        w, h = image_size
        prev_pose[prev_pose[..., 0] < 0] = np.nan
        prev_pose[prev_pose[..., 1] < 0] = np.nan
        prev_pose[prev_pose[..., 0] >= w] = np.nan
        prev_pose[prev_pose[..., 1] >= h] = np.nan

        # apply NMS on the conditions, keeping older tracks
        order = None
        if self._ctd_track_ages is not None:
            ordering = self._ctd_track_ages.copy()

            # sort by track age, then score
            vis = np.sum(np.all(~np.isnan(prev_pose), axis=-1), axis=-1) > 1
            scores = np.nanmean(prev_pose[vis, :, 2], axis=-1)
            ordering[vis] += scores

            # only keep non-zero scores
            order = ordering.argsort()[::-1]
            order = order[ordering[order] > 0]

        nms_mask = nms.nms_oks(
            prev_pose,
            oks_threshold=self.tracking.threshold_nms,
            oks_sigmas=0.1,
            oks_margin=1.0,
            score_threshold=self.tracking.threshold_ctd,
            order=order,
        )

        # Set the previous pose and ID ordering
        if np.any(nms_mask):
            self._prev_pose = prev_pose[nms_mask]

            # get the IDs of the kept poses
            found_idx_to_id = np.where(nms_mask)[0]
            missing_ids = np.where(~nms_mask)[0]
            self._idx_to_id = np.concatenate([found_idx_to_id, missing_ids])

            # add 1 to the age of kept tracks
            if self._ctd_track_ages is None:
                self._ctd_track_ages = np.zeros(len(self._idx_to_id))
            self._ctd_track_ages[nms_mask] += 1
            self._ctd_track_ages[~nms_mask] = 0

            # check if there are any missing individuals
            self._missing_idvs = len(self._prev_pose) != len(self._idx_to_id)
        else:
            self._prev_pose = None
            self._idx_to_id = None
            self._idx_ages = None

    def _merge_conditions(self, bu_cond: np.ndarray) -> np.ndarray:
        """
        Merges conditions made by a BU model with existing conditions from CTD tracking.
        """
        # prepare the BU conditions for matching
        bu_cond = bu_cond.copy()[:, :, :3]
        # mask low-quality keypoints
        bu_cond[bu_cond[..., 2] < self.tracking.threshold_ctd] = np.nan

        # remove non-visible individuals
        kpt_vis = np.all(~np.isnan(bu_cond), axis=-1)
        idv_vis = np.sum(kpt_vis, axis=-1) > 1  # need at least 2 kpts for OKS

        # if no valid BU predictions are left, return the CTD conditions
        if np.sum(idv_vis) == 0:
            return self._prev_pose

        # match BU conditions to CTD poses from the highest score to the lowest
        bu_cond = bu_cond[idv_vis]
        new_conditions = []
        for bu_pose in bu_cond:
            best_oks = 0
            for ctd_pose in self._prev_pose:
                best_oks = max(
                    best_oks,
                    calc_object_keypoint_similarity(bu_pose, ctd_pose, sigma=0.1),
                )

            if best_oks < self.tracking.threshold_bu_add:
                new_conditions.append((best_oks, bu_pose))

        # add the conditions with the lowest OKS score
        new_conditions = [c[1] for c in sorted(new_conditions, key=lambda x: x[0])]

        # if there are no new conditions,
        if len(new_conditions) == 0:
            return self._prev_pose

        new_conditions = np.stack(new_conditions, axis=0)
        cond_pose = np.concatenate([self._prev_pose, new_conditions], axis=0)
        return cond_pose[: len(self._idx_to_id)]


class DetectorInferenceRunner(InferenceRunner[BaseDetector]):
    """Runner for object detection inference"""

    def __init__(self, model: BaseDetector, **kwargs):
        """
        Args:
            model: The detector to use for inference.
            **kwargs: Inference runner kwargs.
        """
        super().__init__(model, **kwargs)

    def predict(
        self, inputs: torch.Tensor, **kwargs
    ) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head, e.g.
            [
                {
                    "bodypart": {"poses": np.ndarray},
                    "unique_bodypart": "poses": np.ndarray},
                }
            ]
        """
        if self.device and "cuda" in str(self.device):
            with torch.autocast(device_type=str(self.device)):
                _, raw_predictions = self.model(inputs.to(self.device))
        else:
            _, raw_predictions = self.model(inputs.to(self.device))
        
        predictions = []
        for item in raw_predictions:
            if isinstance(item, dict) and "boxes" in item and "scores" in item:
                predictions.append({
                    "detection": {
                        "bboxes": item["boxes"].cpu().numpy().reshape(-1, 4),
                        "bbox_scores": item["scores"].cpu().numpy().reshape(-1),
                    }
                })
            else:
                # Handle unexpected output format
                predictions.append({
                    "detection": {
                        "bboxes": np.zeros((0, 4)),
                        "bbox_scores": np.zeros(0),
                    }
                })
        
        return predictions
    
    def inference(self, images) -> list[dict[str, np.ndarray]]:
        """Run inference using the detector's own inference method if available
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays
            
        Returns:
            List of detection results with bboxes in xywh format
        """
        # Use the detector's own inference method if it exists
        if hasattr(self.model, 'inference'):
            return self.model.inference(images)
        else:
            # Fall back to standard inference pipeline
            return super().inference(images)


class TorchvisionDetectorInferenceRunner(DetectorInferenceRunner):
    """Runner for torchvision detector inference that bypasses standard preprocessing"""
    
    def __init__(self, model: BaseDetector, **kwargs):
        """
        Args:
            model: The torchvision detector to use for inference.
            **kwargs: Inference runner kwargs.
        """
        super().__init__(model, **kwargs)
        
    def predict(
        self, inputs: torch.Tensor, **kwargs
    ) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            inputs: the inputs to the model, of shape (batch_size, ...)

        Returns:
            predictions for each of the 'batch_size' inputs, made by each head
        """
        if self.device and "cuda" in str(self.device):
            with torch.autocast(device_type=str(self.device)):
                _, raw_predictions = self.model(inputs.to(self.device))
        else:
            _, raw_predictions = self.model(inputs.to(self.device))
        
        predictions = []
        for item in raw_predictions:
            if isinstance(item, dict) and "boxes" in item:
                predictions.append({
                    "detection": {
                        "bboxes": item["boxes"].cpu().numpy().reshape(-1, 4),
                        "bbox_scores": item["scores"].cpu().numpy().reshape(-1),
                    }
                })
            else:
                # Handle unexpected output format
                predictions.append({
                    "detection": {
                        "bboxes": np.zeros((0, 4)),
                        "bbox_scores": np.zeros(0),
                    }
                })
        
        return predictions
        
    def inference(self, images) -> list[dict[str, np.ndarray]]:
        """Run inference using the torchvision detector's inference method
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays
            
        Returns:
            List of detection results with bboxes in xywh format
        """
        # Always use the detector's own inference method for torchvision detectors
        if hasattr(self.model, 'inference'):
            return self.model.inference(images)
        else:
            # This should never happen for torchvision detectors
            raise RuntimeError("TorchvisionDetectorInferenceRunner requires model to have inference method")


def build_inference_runner(
    task: Task,
    model: nn.Module,
    device: str,
    snapshot_path: str | Path | None = None,
    batch_size: int = 1,
    preprocessor: Preprocessor | None = None,
    postprocessor: Postprocessor | None = None,
    dynamic: DynamicCropper | None = None,
    load_weights_only: bool | None = None,
    async_mode: bool = True,
    num_prefetch_batches: int = 4,
    timeout: float = 30.0,
    **kwargs,
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
        dynamic: The DynamicCropper used for video inference, or None if dynamic
            cropping should not be used. Only for bottom-up pose estimation models.
            Should only be used when creating inference runners for video pose
            estimation with batch size 1.
        load_weights_only: Value for the torch.load() `weights_only` parameter.
            If False, the python pickle module is used implicitly, which is known to
            be insecure. Only set to False if you're loading data that you trust (e.g.
            snapshots that you created). For more information, see:
                https://pytorch.org/docs/stable/generated/torch.load.html
            If None, the default value is used:
                `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`
        async_mode: Whether to use async inference with pipeline parallelism
        num_prefetch_batches: Number of batches to prefetch in async mode
        timeout: Timeout for queue operations in async mode
        **kwargs: Other arguments for the InferenceRunner.

    Returns:
        The inference runner.
    """
    kwargs = dict(
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        load_weights_only=load_weights_only,
        async_mode=async_mode,
        num_prefetch_batches=num_prefetch_batches,
        timeout=timeout,
        **kwargs,
    )

    if task == Task.DETECT:
        if dynamic is not None:
            raise ValueError(
                f"The DynamicCropper can only be used for pose estimation; not object "
                f"detection. Please turn off dynamic cropping."
            )
        
        # Simple check: if superanimal_humanbody, use torchvision inference
        # Otherwise, use standard inference
        if hasattr(model, 'superanimal_name') and model.superanimal_name == "superanimal_humanbody":
            return TorchvisionDetectorInferenceRunner(**kwargs)
        else:
            return DetectorInferenceRunner(**kwargs)

    if task != Task.BOTTOM_UP:
        if dynamic is not None and not isinstance(dynamic, TopDownDynamicCropper):
            print(
                "Turning off dynamic cropping. It should only be used for bottom-up "
                f"pose estimation models, but you are using a {task} model. To use "
                f"dynamic cropping with {task}, use a TopDownDynamicCropper."
            )
            dynamic = None

    if task == Task.COND_TOP_DOWN:
        return CTDInferenceRunner(**kwargs)

    return PoseInferenceRunner(dynamic=dynamic, **kwargs)
