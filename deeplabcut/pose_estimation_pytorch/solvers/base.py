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
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import deeplabcut.pose_estimation_pytorch.data.dataset as deeplabcut_pose_estimation_pytorch_data_dataset
import deeplabcut.pose_estimation_pytorch.models.model as deeplabcut_pose_estimation_pytorch_models_model
import deeplabcut.pose_estimation_pytorch.models.predictors as deeplabcut_pose_estimation_pytorch_models_predictors
import deeplabcut.pose_estimation_pytorch.solvers.inference as deeplabcut_pose_estimation_pytorch_solvers_inference
import deeplabcut.pose_estimation_pytorch.solvers.utils as solver_utils
from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
from deeplabcut.pose_estimation_pytorch.solvers.logger import BaseLogger

SOLVERS = Registry("solvers", build_func=build_from_cfg)


class Solver(ABC):
    """Solver base class.

    Contains helper methods for bundling a model, criterion and optimizer.
    """

    def __init__(
        self,
        model: deeplabcut_pose_estimation_pytorch_models_model.PoseModel,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        predictor: deeplabcut_pose_estimation_pytorch_models_predictors.BasePredictor,
        cfg: Dict,
        device: str = "cpu",
        snapshot_path: Optional[str] = "",
        scheduler: torch.optim.lr_scheduler = None,
        logger: Optional[BaseLogger] = None,
    ):
        """Constructor of the Solver class.

        Args:
            model: The neural network for solving pose estimation task.
            criterion: The criterion computed from the difference
                                  between the prediction and the target.
            optimizer: A PyTorch optimizer for
                                               updating model parameters.
            cfg: DeepLabCut pose_cfg for training.
            See https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_cfg.yaml
                        for more details.
            device: str representing the device on which a torch.Tensor
                          is or will be allocated.
                          Possible value are: ('cpu', 'cuda' or 'mps')
            snapshot_path: path of the snapshot/weights file to load before training
            scheduler: Scheduler for adjusting the
                            lr of the optimizer.
            logger: logger to monitor training (e.g WandB logger)

        Returns:
            None

        Notes/TODO:
            Read stride from config file
        """

        if cfg is None:
            raise ValueError("")
        self.model = model
        self.device = device
        self.cfg = cfg
        self.model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.predictor = predictor
        self.history = {"train_loss": [], "eval_loss": []}
        self.logger = logger
        self.starting_epoch = 0
        if self.logger:
            logger.log_config(cfg)
        if snapshot_path:
            snapshot = torch.load(snapshot_path)
            self.model.load_state_dict(snapshot["model_state_dict"])
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self.starting_epoch = snapshot["epoch"]
        self.stride = 8

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        train_fraction: float = 0.95,
        shuffle: int = 0,
        model_prefix: str = "",
        *,
        epochs: int = 10000,
    ) -> None:
        """Train model for the specified number of steps.

        Args:
            train_loader: Data loader, which is an iterator over train instances.
                          Each batch contains image tensor and heat maps tensor input samples.
            valid_loader: Data loader used for validation of the model.
            train_fraction: Fraction used for training. Defaults to 0.95.
            shuffle: Shuffle id to use from the randomized training sets. Defaults to 0.
            model_prefix: Defaults to "".
            epochs: The number of training epochs. Defaults to 10000.

        Returns:
            None

        Example:
           solver = Solver(model, criterion, optimizer, predictor, cfg, device='cuda')
           solver.fit(train_loader, valid_loader, epochs=50)
        """

        model_folder = solver_utils.get_model_folder(
            train_fraction, shuffle, model_prefix, train_loader.dataset.cfg
        )

        for i in range(self.starting_epoch, epochs):
            train_loss = self.epoch(train_loader, mode="train", step=i + 1)
            if self.scheduler:
                self.scheduler.step()
            logging.info(
                f"Training for epoch {i + 1} done, starting eval on validation data"
            )
            valid_loss = self.epoch(valid_loader, mode="eval", step=i + 1)

            if (i + 1) % self.cfg["save_epochs"] == 0:
                logging.info(f"Finished epoch {i + 1}; saving model")
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": i + 1,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "validation_loss": valid_loss,
                    },
                    f"{model_folder}/train/snapshot-{i + 1}.pt",
                )

            logging.info(
                f"Epoch {i + 1}/{epochs}, "
                f"train loss {float(train_loss):.5f}, "
                f"valid loss {float(valid_loss):.5f}, "
                f'lr {self.optimizer.param_groups[0]["lr"]}'
            )

    def epoch(
        self,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
        step: Optional[int] = None,
    ) -> float:
        """Facilitates training over an epoch. Returns the loss over the batches.

        Args:
            loader: Data loader, which is an iterator over instances.
                                                  Each batch contains image tensor and heat maps tensor input samples.
            mode: str identifier to instruct the Solver whether to train or evaluate.
                        Possible values are: "train" or "eval".
                        Defaults to "train".
            step: the global step in processing, used to log metrics. Defaults to None.

        Raises:
            ValueError: "Solver mode must be train or eval, found mode={mode}."
                        This error is raised when the given mode is invalid (eg. not train nor eval)

        Returns:
            epoch_loss: Average of the loss over the batches.
        """

        if mode not in ["train", "eval"]:
            raise ValueError(f"Solver mode must be train or eval, found mode={mode}.")
        to_mode = getattr(self.model, mode)
        to_mode()
        epoch_loss = []
        metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            losses_dict = self.step(batch, mode)
            epoch_loss.append(losses_dict["total_loss"])

            for key in losses_dict.keys():
                metrics[key].append(losses_dict[key])

            if (i + 1) % self.cfg["display_iters"] == 0:
                logging.info(
                    f"Number of iterations: {i+1}, "
                    f"loss: {losses_dict['total_loss']:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']}"
                )
        epoch_loss = np.mean(epoch_loss).item()
        self.history[f"{mode}_loss"].append(epoch_loss)

        if self.logger:
            for key in metrics:
                self.logger.log(
                    f"{mode} {key}",
                    np.nanmean(metrics[key]),
                    step=step,
                )

        return epoch_loss

    @abstractmethod
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def inference(
        self, dataset: deeplabcut_pose_estimation_pytorch_data_dataset.PoseDataset
    ) -> np.ndarray:
        """Run inference on the given dataset and obtain predicted poses.

        Args:
            dataset: The dataset  to run the inference over

        Returns:
            Numpy array containing the predicted poses

        Notes/TODO:
            Add scale
        """

        predicted_poses = []
        for item in dataset:
            if isinstance(item, tuple) or isinstance(item, list):
                item = item[0]
            item = item.to(self.device)
            output = self.model(item)
            pose = deeplabcut_pose_estimation_pytorch_solvers_inference.get_prediction(
                self.cfg, output, self.stride
            )
            predicted_poses.append(pose)
        predicted_poses = np.concatenate(predicted_poses)
        return predicted_poses


class BottomUpSolver(Solver):

    """Base solvers for bottom up pose estimation."""

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str = "train"
    ) -> dict:
        """Perform a single epoch gradient update or validation step.

        Args:
            batch: Tuple of input image(s) and target(s) for train or valid single step.
            mode: `train` or `eval`. Defaults to "train".

        Raises:
            ValueError: "Solver must be in train or eval mode, but {mode} was found."

        Returns
        -------
        dict : {
            'batch loss' : torch.Tensor,
            'heatmap_loss' : torch.Tensor,
            'locref_loss' : torch.Tensor
        }
        """

        if mode not in ["train", "eval"]:
            raise ValueError(
                f"Solver must be in train or eval mode, but {mode} was found."
            )
        if mode == "train":
            self.optimizer.zero_grad()
        image = batch["image"]
        image = image.to(self.device)
        prediction = self.model(image)

        target = self.model.get_target(
            batch["annotations"], prediction, image.shape[2:]
        )  # (batch_size, channels, h, w)
        for key in target:
            if target[key] is not None:
                target[key] = torch.Tensor(target[key]).to(self.device)

        losses_dict = self.criterion(prediction, target)
        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        for key in losses_dict.keys():
            losses_dict[key] = losses_dict[key].detach().cpu().numpy()
        return losses_dict
