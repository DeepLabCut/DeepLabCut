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

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.solvers.base import SOLVERS, Solver
from deeplabcut.pose_estimation_pytorch.solvers.utils import *


@SOLVERS.register_module
class TopDownSolver(Solver):
    """Top Down Solver Class

    This class is used for training the top-down pose estimation model
    based on a detector model such as FasterRCNN.
    Only supports FasterRCNN as a detector for now (https://github.com/rbgirshick/fast-rcnn)

    Attributes:
        detector: The detector model used in the top-down approach.
        detector_optimizer: Optimizer for the detector model.
        detector_criterion: Criterion for the detector model (not used with FasterRCNN).
        detector_scheduler: Scheduler for the detector model (optional).
        detector_path: Path to a pre-trained detector model checkpoint (optional).

    Examples:
        # Initialize the top-down solver with a FasterRCNN detector and its optimizer
        detector = FasterRCNN()
        detector_optimizer = torch.optim.SGD(detector.parameters(), lr=0.001, momentum=0.9)
        solver = TopDownSolver(detector=detector, detector_optimizer=detector_optimizer)

        # Load data loaders for training and validation
        train_detector_loader = torch.utils.data.DataLoader(...)
        valid_detector_loader = torch.utils.data.DataLoader(...)
        train_pose_loader = torch.utils.data.DataLoader(...)
        valid_pose_loader = torch.utils.data.DataLoader(...)

        # Train the top-down pose estimation model
        solver.fit(
            train_detector_loader=train_detector_loader,
            valid_detector_loader=valid_detector_loader,
            train_pose_loader=train_pose_loader,
            valid_pose_loader=valid_pose_loader,
            train_fraction=0.95,
            shuffle=0,
            model_prefix="my_model",
            epochs=10000
        )
    """

    def __init__(
        self,
        *args,
        detector: BaseDetector,
        detector_optimizer: torch.optim.Optimizer,
        detector_criterion: nn.Module = None,  # Not Used with fasterRCNN
        detector_scheduler: Optional = None,
        detector_path: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detector = detector
        self.detector.to(self.device)
        self.detector_optimizer = detector_optimizer
        self.detector_criterion = detector_criterion
        self.detector_scheduler = detector_scheduler
        self.starting_epoch = 0
        self.starting_epoch_detector = 0
        if detector_path:
            detector = torch.load(detector_path)
            self.detector.load_state_dict(detector["detector_state_dict"])
            self.detector_optimizer.load_state_dict(detector["optimizer_state_dict"])
            self.starting_epoch_detector = detector["epoch"]

    def fit(
        self,
        train_detector_loader: torch.utils.data.DataLoader,
        valid_detector_loader: torch.utils.data.DataLoader,
        train_pose_loader: torch.utils.data.DataLoader,
        valid_pose_loader: torch.utils.data.DataLoader,
        train_fraction: float = 0.95,
        shuffle: int = 0,
        model_prefix: str = "",
        *,
        epochs: int = 10000,
        detector_epochs: int = 10000,
    ):
        model_folder = get_model_folder(
            train_fraction, shuffle, model_prefix, train_detector_loader.dataset.cfg
        )

        for i in range(self.starting_epoch_detector, detector_epochs):
            train_detector_loss = self.epoch_detector(
                train_detector_loader, mode="train", step=i + 1
            )
            if self.detector_scheduler:
                self.detector_scheduler.step()
            print(f"Training the detector for epoch {i + 1} done")

            # TODO no eval pass for the detector since fasterRCNN can't return a loss in eval mode

            if (i + 1) % self.cfg["detector"].get("detector_save_epochs", 1) == 0:
                print(f"Finished epoch {i + 1}; saving detector")
                torch.save(
                    {
                        "detector_state_dict": self.detector.state_dict(),
                        "epoch": i + 1,
                        "optimizer_state_dict": self.detector_optimizer.state_dict(),
                        "train_loss": train_detector_loss,
                    },
                    f"{model_folder}/train/detector-snapshot-{i + 1}.pt",
                )
            print(
                f"Epoch {i + 1}/{detector_epochs}, "
                f"train detector loss {train_detector_loss:.5f}"
            )

        if detector_epochs % self.cfg["detector"].get("detector_save_epochs", 1) != 0:
            torch.save(
                self.detector.state_dict(),
                f"{model_folder}/train/detector-snapshot-{epochs}.pt",
            )
            print(f"Finished epoch {detector_epochs}; saving model")

        for i in range(self.starting_epoch, epochs):
            train_pose_loss = self.epoch_pose(
                train_pose_loader, mode="train", step=i + 1
            )
            if self.scheduler:
                self.scheduler.step()

            print(
                f"Training the pose estimator for epoch {i + 1} done, starting eval on validation data"
            )

            valid_pose_loss = self.epoch_pose(
                valid_pose_loader, mode="eval", step=i + 1
            )

            if (i + 1) % self.cfg["save_epochs"] == 0:
                print(f"Finished epoch {i + 1}; saving pose model")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": i + 1,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_pose_loss,
                        "validation_loss": valid_pose_loss,
                    },
                    f"{model_folder}/train/snapshot-{i + 1}.pt",
                )

            print(
                f"Epoch {i + 1}/{epochs}, "
                f"train pose loss {train_pose_loss:.5f}, "
                f"valid pose loss {valid_pose_loss:.5f}"
            )

        if epochs % self.cfg["save_epochs"] != 0:
            print(f"Finished epoch {epochs}; saving model")
            torch.save(
                self.model.state_dict(),
                f"{model_folder}/train/pose-snapshot-{epochs}.pt",
            )

    def epoch(self, *args):
        # Unused in top down since we are dealing with two different epoch functions
        pass

    def step(self, *args):
        """Unused in top down since we are dealing with two different step functions."""
        pass

    def epoch_detector(
        self,
        detector_loader: torch.utils.data.DataLoader,
        mode: str = "train",
        step: Optional[int] = None,
    ) -> float:
        if mode not in ["train", "eval"]:
            raise ValueError(f"Solver mode must be train or eval, found mode={mode}.")
        to_mode_detector = getattr(self.detector, mode)
        to_mode_detector()
        epoch_detector_loss = []
        metrics = defaultdict(list)

        # Detector training
        for i, batch_d in enumerate(detector_loader):
            detector_loss = self.step_detector(batch_d, mode)
            epoch_detector_loss.append(detector_loss)

            metrics["detector_loss"].append(detector_loss)

            # TODO good for evaluation speed up but should be optional
            # if mode == "eval" and i > 100:
            #     break

            if (i + 1) % self.cfg["display_iters"] == 0:
                print(
                    f"Number of iterations for detector: {i+1}, loss : {np.mean(metrics['detector_loss']):.5f}, lr : {self.optimizer.param_groups[0]['lr']}"
                )
        epoch_detector_loss = np.mean(epoch_detector_loss)

        # TODO is history really necessary here ?
        # self.history[f'{mode}_loss'].append(epoch_loss)

        if self.logger:
            for key in metrics.keys():
                self.logger.log(
                    f"{mode} {key}",
                    np.nanmean(metrics[key]),
                    step=step,
                )

        return epoch_detector_loss

    def epoch_pose(
        self,
        pose_loader: torch.utils.data.DataLoader,
        mode: str = "train",
        step: Optional[int] = None,
    ) -> float:
        if mode not in ["train", "eval"]:
            raise ValueError(f"Solver mode must be train or eval, found mode={mode}.")
        to_mode_pose = getattr(self.model, mode)
        to_mode_pose()
        epoch_pose_loss = []
        metrics = defaultdict(list)

        # Pose model training
        for i, batch in enumerate(pose_loader):
            losses_dict = self.step_pose(batch, mode)
            epoch_pose_loss.append(losses_dict["total_loss"])

            for key in losses_dict.keys():
                metrics["pose_" + key].append(losses_dict[key])

            # TODO good for evaluation speed up but should be optional
            # if mode == "eval" and i > 100:
            #     break

            if (i + 1) % self.cfg["display_iters"] == 0:
                print(
                    f"Number of iterations for pose: {i+1}, loss : {np.mean(metrics['pose_total_loss']):.5f}, lr : {self.optimizer.param_groups[0]['lr']}"
                )
        epoch_pose_loss = np.mean(epoch_pose_loss)

        # TODO is history really necessary here ?
        # self.history[f'{mode}_loss'].append(epoch_loss)

        if self.logger:
            for key in metrics.keys():
                self.logger.log(
                    f"{mode} {key}",
                    np.nanmean(metrics[key]),
                    step=step,
                )

        return epoch_pose_loss

    def step_detector(self, batch: dict, mode: str = "train") -> float:
        """Performs a step for the detector over a batch

        Args:
            batch: batch returned by the dataloader
            mode: "train" or "eval". Defaults to "train".

        Returns:
            loss : loss for the detector
        """
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"Solver must be in train or eval mode, but {mode} was found."
            )
        if mode == "train":
            self.detector_optimizer.zero_grad()

        images = batch["image"]
        images = images.to(self.device)

        target = self.detector.get_target(
            batch["annotations"]
        )  # (batch_size, channels, h, w)
        for item in target:  # target is a list here
            for key in item:
                if item[key] is not None:
                    item[key] = torch.tensor(item[key]).to(self.device)

        if mode == "train":
            # For now only FasterRCNN is supported and it already returns the loss dict
            # when calling forward()
            losses_dict = self.detector(images, target)
            loss = sum(l for l in losses_dict.values())

            loss.backward()
            self.detector_optimizer.step()

            return loss.detach().cpu().numpy()
        else:
            # No way to get losses in eval mode for the moment
            return 0.0

    def step_pose(self, batch: dict, mode: str = "train") -> dict:
        if mode not in ["train", "eval"]:
            raise ValueError(
                f"Solver must be in train or eval mode, but {mode} was found."
            )
        if mode == "train":
            self.optimizer.zero_grad()

        images = batch["image"]
        images = images.to(self.device)

        prediction = self.model(images)

        target = self.model.get_target(
            batch["annotations"], prediction, images.shape[2:]
        )  # (batch_size, channels, h, w)
        for key in target:
            if target[key] is not None:
                target[key] = torch.tensor(target[key]).to(self.device)

        losses_dict = self.criterion(prediction, target)
        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        for key in losses_dict.keys():
            losses_dict[key] = losses_dict[key].detach().cpu().numpy()
        return losses_dict
