from typing import Optional
from typing import Tuple, Dict
import torch
import torch.nn as nn
import numpy as np

from deeplabcut.pose_estimation_pytorch.solvers.base import Solver, SOLVERS
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.solvers.utils import *


@SOLVERS.register_module
class TopDownSolver(Solver):
    """
    Top down solver

    Currently very specific to FasterRCNN for detectpr since torchvison's implementation isn't flexible
    """

    def __init__(
        self,
        *args,
        detector: BaseDetector,
        detector_optimizer: torch.optim.Optimizer,
        detector_criterion: nn.Module = None,  # Not Used with fasterRCNN
        detector_scheduler: Optional = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detector = detector
        self.detector_optimizer = detector_optimizer
        self.detector_criterion = detector_criterion
        self.detector_scheduler = detector_scheduler
        self.detector.to(self.device)

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
    ):
        """
        Train model for the specified number of steps.

        Parameters
        ----------
        train_detector_loader: Data loader, which is an iterator over train instances.
            Each batch contains image tensor and heat maps tensor input samples.
        valid_detector_loader: Data loader used for validation of the detector model.
        train_pose_loader: Data loader used for the pose detection part of the top down model
        valid_pose_loader: Data loader used for validaton of the pose regression part of the top down model
        train_fraction: TODO discuss (mb better specify with config)
        shuffle: TODO discuss (mb better specify with config)
        model_prefix: TODO discuss (mb better specify with config)
        epochs: The number of training iterations.
        """
        model_folder = get_model_folder(
            train_fraction, shuffle, model_prefix, train_detector_loader.dataset.cfg
        )

        for i in range(epochs):
            train_detector_loss, train_pose_loss = self.epoch(
                train_detector_loader, train_pose_loader, mode="train", step=i + 1
            )
            if self.scheduler:
                self.scheduler.step()
            if self.detector_scheduler:
                self.detector_scheduler.step()
            print(f"Training for epoch {i + 1} done, starting eval on validation data")
            valid_detector_loss, valid_pose_loss = self.epoch(
                valid_detector_loader, valid_pose_loader, mode="eval", step=i + 1
            )

            if (i + 1) % self.cfg["save_epochs"] == 0:
                print(f"Finished epoch {i + 1}; saving model")
                torch.save(
                    self.model.state_dict(),
                    f"{model_folder}/train/snapshot-{i + 1}.pt",
                )
                torch.save(
                    self.detector.state_dict(),
                    f"{model_folder}/train/detector-snapshot-{i + 1}.pt",
                )

            print(
                f"Epoch {i + 1}/{epochs}, "
                f"train detector loss {train_detector_loss}, "
                f"valid detector loss {valid_detector_loss}"
                f"train pose loss {train_pose_loss}"
                f"valid pose loss {valid_pose_loss}"
            )

        if epochs % self.cfg["save_epochs"] != 0:
            print(f"Finished epoch {epochs}; saving model")
            torch.save(
                self.model.state_dict(),
                f"{model_folder}/train/pose-snapshot-{epochs}.pt",
            )
            torch.save(
                self.detector.state_dict(),
                f"{model_folder}/train/detector-snapshot-{epochs}.pt",
            )

    def step(self, *args):
        # Unused in top down since we are dealing with two different step functions
        pass

    def epoch(
        self,
        detector_loader: torch.utils.data.DataLoader,
        pose_loader: torch.utils.data.DataLoader,
        mode: str = "train",
        step: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        detector_loader: Data loader, which is an iterator over instances.
            Each batch contains image tensors.
        pose_loader: Data loader, Each batch contains a cropped image around an animal
        mode: "train" or "eval"
        step: the global step in processing, used to log metrics.
        Returns
        -------
        epoch_loss: Average of the loss over the batches.
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"Solver mode must be train or eval, found mode={mode}.")
        to_mode_pose = getattr(self.model, mode)
        to_mode_pose()
        to_mode_detector = getattr(self.detector, mode)
        to_mode_detector()
        epoch_detector_loss, epoch_pose_loss = [], []
        metrics = {
            "total_pose_loss": [],
            "detector_loss": [],
        }

        # Pose model training
        for i, batch in enumerate(pose_loader):
            total_loss = self.step_pose(batch, mode)
            epoch_pose_loss.append(total_loss)

            metrics["total_pose_loss"].append(total_loss)

            if mode == "eval" and i > 100:
                break

            if (i + 1) % self.cfg["display_iters"] == 0:
                print(
                    f"Number of iterations for pose: {i+1}, loss : {np.mean(metrics['total_pose_loss'])}, lr : {self.optimizer.param_groups[0]['lr']}"
                )
        epoch_pose_loss = np.mean(epoch_pose_loss)

        # Detector training
        for i, batch_d in enumerate(detector_loader):
            detector_loss = self.step_detector(batch_d, mode)
            epoch_detector_loss.append(detector_loss)

            metrics["detector_loss"].append(detector_loss)

            if mode == "eval" and i > 100:
                break

            if (i + 1) % self.cfg["display_iters"] == 0:
                print(
                    f"Number of iterations for detector: {i+1}, loss : {np.mean(metrics['detector_loss'])}, lr : {self.optimizer.param_groups[0]['lr']}"
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

        return epoch_detector_loss, epoch_pose_loss

    def step_detector(self, batch, mode: str = "train"):
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

    def step_pose(self, batch, mode: str = "train"):
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

        total_loss = self.criterion(prediction, target)
        if mode == "train":
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach().cpu().numpy()
