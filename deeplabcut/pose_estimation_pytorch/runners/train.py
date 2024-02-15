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

import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Generic

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import ModelType, Runner, Task
from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger


class TrainingRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
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
        save_optimizer_state: bool = False,
    ):
        """
        Args:
            model: the model to run actions on
            optimizer: the optimizer to use when fitting the model
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            snapshot_prefix: the prefix with which to save snapshots
            snapshot_path: if defined, the path of a snapshot from which to load pretrained weights
            scheduler: Scheduler for adjusting the lr of the optimizer.
            logger: logger to monitor training (e.g WandB logger)
            save_optimizer_state: whether to save the optimizer state, which allows to
                restart training (warning - this makes the snapshots much heavier)
        """
        super().__init__(model=model, device=device, snapshot_path=snapshot_path)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history: dict[str, list] = {"train_loss": [], "eval_loss": []}
        self.snapshot_prefix = snapshot_prefix
        self.logger = logger
        self.starting_epoch = 0
        self.save_optimizer_state = save_optimizer_state

        if self.snapshot_path is not None and len(self.snapshot_path) > 0:
            self.starting_epoch = self.load_snapshot(
                self.snapshot_path,
                self.device,
                self.model,
                self.optimizer,
            )

    @abstractmethod
    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step

        Args:
            batch: the batch data on which to run a step
            mode: "train" or "eval". Defaults to "train".

        Raises:
            ValueError: if mode is not in {"train", "eval"}

        Returns:
            A dictionary containing the different losses for the step
        """

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_folder: str,
        epochs: int,
        save_epochs: int,
        display_iters: int,
    ) -> None:
        """Train model for the specified number of steps.

        Args:
            train_loader: Data loader, which is an iterator over train instances.
                Each batch contains image tensor and heat maps tensor input samples.
            valid_loader: Data loader used for validation of the model.
            model_folder: The folder to which logs should be written and snapshots saved
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
                train_loader, mode="train", step=i + 1, display_iters=display_iters
            )
            if self.scheduler:
                self.scheduler.step()

            logging.info(
                f"Training for epoch {i + 1} done, starting eval on validation data"
            )
            valid_loss = self._epoch(
                valid_loader, mode="eval", step=i + 1, display_iters=display_iters
            )

            if (i + 1) % save_epochs == 0 or (i + 1) == epochs:
                logging.info(f"Finished epoch {i + 1}; saving model")
                save_path = f"{model_folder}/train/{self.snapshot_prefix}-{i + 1}.pt"
                state = {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": i + 1,
                    "train_loss": train_loss,
                    "validation_loss": valid_loss,
                }
                if self.save_optimizer_state:
                    state["optimizer_state_dict"] = self.optimizer.state_dict()
                torch.save(state, save_path)

            logging.info(
                f"Epoch {i + 1}/{epochs}, "
                f"train loss {float(train_loss):.5f}, "
                f"valid loss {float(valid_loss):.5f}, "
                f'lr {self.optimizer.param_groups[0]["lr"]}'
            )

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
                    f"{mode} {key}", np.nanmean(metrics[key]).item(), step=step
                )

        return epoch_loss


class PoseTrainingRunner(TrainingRunner[PoseModel]):
    """Runner to train pose estimation models"""

    def __init__(
        self, model: PoseModel, optimizer: torch.optim.Optimizer, **kwargs
    ):
        """
        Args:
            model: The neural network for solving pose estimation task.
            optimizer: A PyTorch optimizer for updating model parameters.
            **kwargs: TrainingRunner kwargs
        """
        super().__init__(model, optimizer, **kwargs)

    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                "total_loss": aggregate_loss,
                "aux_loss_1": loss_value,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"BottomUpSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()

        batch_inputs = batch["image"]
        batch_inputs = batch_inputs.to(self.device)
        head_outputs = self.model(batch_inputs)

        target = self.model.get_target(batch_inputs, head_outputs, batch["annotations"])

        losses_dict = self.model.get_loss(head_outputs, target)
        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        return {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}


class DetectorTrainingRunner(TrainingRunner[BaseDetector]):
    """Runner to train object detection models"""

    def __init__(
        self,
        model: BaseDetector,
        optimizer: torch.optim.Optimizer,
        snapshot_prefix: str = "detector-snapshot",
        **kwargs,
    ):
        """
        Args:
            model: The detector model to train.
            optimizer: The optimizer to use to train the model.
            **kwargs: TrainingRunner kwargs
        """
        super().__init__(model, optimizer, snapshot_prefix=snapshot_prefix, **kwargs)

    def step(
        self, batch: dict[str, Any], mode: str = "train"
    ) -> dict[str, torch.Tensor]:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Runner must be in train or eval mode, but {mode} was found."

        Returns:
            dict: {
                'total_loss': torch.Tensor,
                'aux_loss_1': torch.Tensor,
                ...,
            }
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"DetectorSolver must be in train or eval mode, but {mode} was found."
            )

        if mode == "train":
            self.optimizer.zero_grad()
        else:
            # Override base class
            # No losses returned in train mode;
            #   see https://stackoverflow.com/a/65347721
            # Should be safe as BN is frozen;
            #   see https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/12
            self.model.train()

        images = batch["image"]
        images = images.to(self.device)

        target = self.model.get_target(
            batch["annotations"]
        )  # (batch_size, channels, h, w)
        for item in target:  # target is a list here
            for key in item:
                if item[key] is not None:
                    item[key] = item[key].to(self.device)

        losses, _ = self.model(images, target)
        losses["total_loss"] = sum(loss_part for loss_part in losses.values())
        if mode == "train":
            losses["total_loss"].backward()
            self.optimizer.step()

        return {k: v.detach().cpu().numpy() for k, v in losses.items()}


def build_training_runner(
    task: Task,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    snapshot_path: str | None = None,
    logger: BaseLogger | None = None,
) -> TrainingRunner:
    """
    Build a runner object according to a pytorch configuration file

    Args:
        task: the task the runner will perform
        model: the model to run
        optimizer: the optimizer to use to train the model
        scheduler: the scheduler to use to train the model
        device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
        snapshot_path: the snapshot from which to load the weights
        logger: the logger to use, if any

    Returns:
        the runner that was built
    """
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        device=device,
        snapshot_path=snapshot_path,
        scheduler=scheduler,
        logger=logger,
    )
    if task == Task.DETECT:
        return DetectorTrainingRunner(**kwargs)

    return PoseTrainingRunner(**kwargs)
