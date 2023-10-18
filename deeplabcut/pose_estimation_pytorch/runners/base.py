#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Generic, TypeVar, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeplabcut.pose_estimation_pytorch.data.preprocessor import Preprocessor
from deeplabcut.pose_estimation_pytorch.data.postprocessor import Postprocessor
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger


RUNNERS = Registry("runners", build_func=build_from_cfg)
ModelType = TypeVar("ModelType", bound=nn.Module)


class Runner(ABC, Generic[ModelType]):
    """Runner base class

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        snapshot_prefix: str = "snapshot",
        snapshot_path: str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        logger: BaseLogger | None = None,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
    ):
        """
        Args:
            model: the model to run actions on
            optimizer: the optimizer to use when fitting the model
            device: one of {'cpu', 'cuda', 'mps'}; the device to use for training/inference
            snapshot_prefix: the prefix with which to save snapshots
            snapshot_path: if defined, the path of a snapshot from which to load pretrained weights
            scheduler: Scheduler for adjusting the lr of the optimizer.
            logger: logger to monitor training (e.g WandB logger)
            preprocessor: the preprocessor to use on images before inference
            postprocessor: the postprocessor to use on images after inference
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history: dict[str, list] = {"train_loss": [], "eval_loss": []}
        self.snapshot_prefix = snapshot_prefix
        self.logger = logger
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.starting_epoch = 0
        if snapshot_path:
            snapshot = torch.load(snapshot_path)
            self.model.load_state_dict(snapshot["model_state_dict"])
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self.starting_epoch = snapshot["epoch"]

    @abstractmethod
    def step(
        self,
        batch: dict[str, Any],
        mode: str = "train",
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step"""

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        """Makes predictions from a model input and output

        Args:
            the inputs to the model, of shape (batch_size, ...)

        Returns:
            the predictions for each of the 'batch_size' inputs
        """

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_folder: str,
        epochs: int,
        save_epochs: int,
        display_iters: int,
        *args,
        **kwargs,
    ) -> None:
        """Train model for the specified number of steps.

        Args:
            train_loader: Data loader, which is an iterator over train instances.
                Each batch contains image tensor and heat maps tensor input samples.
            valid_loader: Data loader used for validation of the model.
            model_folder: the folder to which logs should be written and snapshots saved
            epochs: The number of training epochs.
            save_epochs: The epoch step at which to save models
            display_iters: The number of iterations between each loss print

        Example:
           runner = Runner(model, optimizer, cfg, device='cuda')
           runner.fit(train_loader, valid_loader, "example/models" epochs=50)
        """
        Path(model_folder).mkdir(exist_ok=True, parents=True)
        self.model.to(self.device)

        for i in range(self.starting_epoch, epochs):
            train_loss = self._epoch(
                train_loader,
                mode="train",
                step=i + 1,
                display_iters=display_iters,
            )
            if self.scheduler:
                self.scheduler.step()

            logging.info(
                f"Training for epoch {i + 1} done, starting eval on validation data"
            )
            valid_loss = self._epoch(
                valid_loader, mode="eval", step=i + 1, display_iters=display_iters
            )

            if (i + 1) % save_epochs == 0:
                logging.info(f"Finished epoch {i + 1}; saving model")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": i + 1,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "validation_loss": valid_loss,
                    },
                    f"{model_folder}/train/{self.snapshot_prefix}-{i + 1}.pt",
                )

            logging.info(
                f"Epoch {i + 1}/{epochs}, "
                f"train loss {float(train_loss):.5f}, "
                f"valid loss {float(valid_loss):.5f}, "
                f'lr {self.optimizer.param_groups[0]["lr"]}'
            )

    @torch.no_grad()
    def inference(
        self,
        images: Iterable[str | np.ndarray]
        | Iterable[tuple[str | np.ndarray, dict[str, Any]]],
    ) -> list[dict[str, dict[str, np.ndarray]]]:
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
                image_predictions, _ = self.postprocessor(image_predictions, context)

            results.append(image_predictions)

        return results

    def _epoch(
        self,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
        step: int | None = None,
        display_iters: int = 500,
    ) -> float:
        """Facilitates training over an epoch. Returns the loss over the batches.

        Args:
            loader: Data loader, which is an iterator over instances.
                Each batch contains image tensor and heat maps tensor input samples.
            mode: str identifier to instruct the Runner whether to train or evaluate.
                Possible values are: "train" or "eval".
            step: the global step in processing, used to log metrics. Defaults to None.
            display_iters: the number of iterations between each loss print

        Raises:
            ValueError: When the given mode is invalid

        Returns:
            epoch_loss: Average of the loss over the batches.
        """
        if mode == "train":
            self.model.train()
        elif mode == "eval" or mode == "inference":
            self.model.eval()
        else:
            raise ValueError(f"Runner mode must be train or eval, found mode={mode}.")

        epoch_loss = []
        metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            losses_dict = self.step(batch, mode)
            epoch_loss.append(losses_dict["total_loss"])

            for key in losses_dict.keys():
                metrics[key].append(losses_dict[key])

            if (i + 1) % display_iters == 0:
                logging.info(
                    f"Number of iterations: {i + 1}, "
                    f"loss: {losses_dict['total_loss']:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']}"
                )

        epoch_loss = np.mean(epoch_loss).item()
        self.history[f"{mode}_loss"].append(epoch_loss)

        if self.logger:
            for key in metrics:
                self.logger.log(
                    f"{mode} {key}",
                    np.nanmean(metrics[key]).item(),
                    step=step,
                )

        return epoch_loss
