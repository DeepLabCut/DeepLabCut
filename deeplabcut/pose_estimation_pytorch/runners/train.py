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
from torch.nn.parallel import DataParallel

from deeplabcut.pose_estimation_pytorch.metrics import compute_bbox_metrics
from deeplabcut.pose_estimation_pytorch.metrics.scoring import (
    get_scores,
    pair_predicted_individuals_with_gt,
)
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import ModelType, Runner
from deeplabcut.pose_estimation_pytorch.runners.logger import (
    BaseLogger,
    CSVLogger,
    ImageLoggerMixin,
)
from deeplabcut.pose_estimation_pytorch.runners.schedulers import build_scheduler
from deeplabcut.pose_estimation_pytorch.runners.snapshots import TorchSnapshotManager
from deeplabcut.pose_estimation_pytorch.task import Task


class TrainingRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
    """Runner base class

    A runner takes a model and runs actions on it, such as training or inference
    """

    def __init__(
        self,
        model: ModelType,
        optimizer: torch.optim.Optimizer,
        snapshot_manager: TorchSnapshotManager,
        device: str = "cpu",
        gpus: list[int] | None = None,
        eval_interval: int = 1,
        snapshot_path: Path | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        logger: BaseLogger | None = None,
    ):
        """
        Args:
            model: the model to run actions on
            optimizer: the optimizer to use when fitting the model
            snapshot_manager: the module to use to manage snapshots
            device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
            gpus: the list of GPU indices to use for multi-GPU training
            eval_interval: how often evaluation is run on the test set (in epochs)
            snapshot_path: if defined, the path of a snapshot from which to load
                pretrained weights
            scheduler: scheduler for adjusting the lr of the optimizer
            logger: logger to monitor training (e.g WandB logger)
        """
        super().__init__(
            model=model, device=device, gpus=gpus, snapshot_path=snapshot_path
        )
        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.snapshot_manager = snapshot_manager
        self.history: dict[str, list] = dict(train_loss=[], eval_loss=[])
        self.csv_logger = CSVLogger(train_folder=snapshot_manager.model_folder)
        self.logger = logger
        self.starting_epoch = 0
        self.current_epoch = 0

        # some models cannot compute a validation loss (e.g. detectors)
        self._print_valid_loss = True

        if self.snapshot_path is not None and self.snapshot_path != "":
            self.starting_epoch = self.load_snapshot(
                self.snapshot_path,
                self.device,
                self.model,
                self.optimizer,
            )

        self._metadata = dict(epoch=self.starting_epoch, metrics=dict(), losses=dict())
        self._epoch_ground_truth = {}
        self._epoch_predictions = {}

    def state_dict(self) -> dict:
        return {
            "metadata": self._metadata,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

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

    @abstractmethod
    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Computes the metrics using the data accumulated during an epoch

        Returns:
            A dictionary containing the different losses for the step
        """
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        display_iters: int,
    ) -> None:
        """Train model for the specified number of steps.

        Args:
            train_loader: Data loader, which is an iterator over train instances.
                Each batch contains image tensor and heat maps tensor input samples.
            valid_loader: Data loader used for validation of the model.
            epochs: The number of training epochs.
            display_iters: The number of iterations between each loss print

        Example:
           runner = Runner(model, optimizer, cfg, device='cuda')
           runner.fit(train_loader, valid_loader, "example/models" epochs=50)
        """
        if self.gpus:
            self.model = DataParallel(self.model, device_ids=self.gpus).cuda()
        else:
            self.model.to(self.device)

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.select_images_to_log(train_loader, valid_loader)

        # continuing to train a model: either total epochs or extra epochs
        if self.starting_epoch > epochs:
            epochs = self.starting_epoch + epochs

        for e in range(self.starting_epoch + 1, epochs + 1):
            self.current_epoch = e
            self._metadata["epoch"] = e
            train_loss = self._epoch(
                train_loader, mode="train", display_iters=display_iters
            )
            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {e}/{epochs} (lr={lr}), train loss {float(train_loss):.5f}"
            if e % self.eval_interval == 0:
                with torch.no_grad():
                    logging.info(f"Training for epoch {e} done, starting evaluation")
                    valid_loss = self._epoch(
                        valid_loader, mode="eval", display_iters=display_iters
                    )
                    if self._print_valid_loss:
                        msg += f", valid loss {float(valid_loss):.5f}"

            self.snapshot_manager.update(e, self.state_dict(), last=(e == epochs))
            logging.info(msg)

    def _epoch(
        self,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
        display_iters: int = 500,
    ) -> float:
        """Facilitates training over an epoch. Returns the loss over the batches.

        Args:
            loader: Data loader, which is an iterator over instances.
                Each batch contains image tensor and heat maps tensor input samples.
            mode: str identifier to instruct the Runner whether to train or evaluate.
                Possible values are: "train" or "eval".
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
        loss_metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            losses_dict = self.step(batch, mode)
            epoch_loss.append(losses_dict["total_loss"])

            for key in losses_dict.keys():
                loss_metrics[key].append(losses_dict[key])

            if (i + 1) % display_iters == 0:
                logging.info(
                    f"Number of iterations: {i + 1}, "
                    f"loss: {losses_dict['total_loss']:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']}"
                )

        perf_metrics = None
        if mode == "eval":
            perf_metrics = self._compute_epoch_metrics()
            self._metadata["metrics"] = perf_metrics
            self._epoch_predictions = {}
            self._epoch_ground_truth = {}
            if perf_metrics is not None and len(perf_metrics) > 0:
                logging.info(f"Epoch {self.current_epoch} performance:")
                for name, score in perf_metrics.items():
                    logging.info(f"{name + ':': <20}{score:.3f}")

        epoch_loss = np.mean(epoch_loss).item()
        self.history[f"{mode}_loss"].append(epoch_loss)

        metrics_to_log = {}
        if perf_metrics:
            for name, score in perf_metrics.items():
                if not isinstance(score, (int, float)):
                    score = 0.0
                metrics_to_log[name] = score

        for key in loss_metrics:
            name = f"{mode}.{key}"
            val = np.nan
            if np.sum(~np.isnan(loss_metrics[key])) > 0:
                val = np.nanmean(loss_metrics[key]).item()
            self._metadata["losses"][name] = val
            metrics_to_log[f"losses/{name}"] = val

        self.csv_logger.log(metrics_to_log, step=self.current_epoch)
        if self.logger:
            self.logger.log(metrics_to_log, step=self.current_epoch)

        return epoch_loss


class PoseTrainingRunner(TrainingRunner[PoseModel]):
    """Runner to train pose estimation models"""

    def __init__(self, model: PoseModel, optimizer: torch.optim.Optimizer, **kwargs):
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

        inputs = batch["image"]
        inputs = inputs.to(self.device).float()
        outputs = self.model(inputs)

        if self.gpus:
            underlying_model = self.model.module
        else:
            underlying_model = self.model

        target = underlying_model.get_target(outputs, batch["annotations"])
        losses_dict = underlying_model.get_loss(outputs, target)
        if mode == "train":
            losses_dict["total_loss"].backward()
            self.optimizer.step()

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.log_images(batch, outputs, target, step=self.current_epoch)

        if mode == "eval":
            predictions = {
                name: {k: v.detach().cpu().numpy() for k, v in pred.items()}
                for name, pred in underlying_model.get_predictions(outputs).items()
            }

            ground_truth = batch["annotations"]["keypoints"]
            if batch["annotations"]["with_center_keypoints"][0]:
                ground_truth = ground_truth[..., :-1, :]

            self._update_epoch_predictions(
                name="bodyparts",
                paths=batch["path"],
                gt_keypoints=ground_truth,
                pred_keypoints=predictions["bodypart"]["poses"],
                offsets=batch["offsets"],
                scales=batch["scales"],
            )
            if "unique_bodypart" in predictions:
                self._update_epoch_predictions(
                    name="unique_bodyparts",
                    paths=batch["path"],
                    gt_keypoints=batch["annotations"]["keypoints_unique"],
                    pred_keypoints=predictions["unique_bodypart"]["poses"],
                    offsets=batch["offsets"],
                    scales=batch["scales"],
                )

        return {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}

    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Computes the metrics using the data accumulated during an epoch
        Returns:
            A dictionary containing the different losses for the step
        """
        num_animals = max(
            [len(kpts) for kpts in self._epoch_ground_truth["bodyparts"].values()]
        )
        poses = pair_predicted_individuals_with_gt(
            self._epoch_predictions["bodyparts"], self._epoch_ground_truth["bodyparts"]
        )

        # pad predictions if there are any missing (needed for top-down models)
        gt, pred = {}, {}
        for path, img_gt in self._epoch_ground_truth["bodyparts"].items():
            for kpt_dict, kpts in [(gt, img_gt), (pred, poses[path])]:
                if len(kpts) < num_animals:
                    padded_kpts = -np.ones((num_animals, *kpts.shape[1:]))
                    padded_kpts[: len(kpts)] = kpts
                    kpt_dict[path] = padded_kpts
                else:
                    kpt_dict[path] = kpts

        scores = get_scores(
            poses=pred,
            ground_truth=gt,
            unique_bodypart_poses=self._epoch_predictions.get("unique_bodyparts"),
            unique_bodypart_gt=self._epoch_ground_truth.get("unique_bodyparts"),
            pcutoff=0.6,
        )
        return {f"metrics/test.{metric}": value for metric, value in scores.items()}

    def _update_epoch_predictions(
        self,
        name: str,
        paths: torch.Tensor,
        gt_keypoints: torch.Tensor,
        pred_keypoints: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Updates the stored predictions with a new batch"""
        epoch_gt_metric = self._epoch_ground_truth.get(name, {})
        epoch_metric = self._epoch_predictions.get(name, {})
        assert len(paths) == len(gt_keypoints) == len(pred_keypoints)
        assert len(paths) == len(offsets) == len(scales)
        scales = scales.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()

        for path, gt, pred, scale, offset in zip(
            paths,
            gt_keypoints,
            pred_keypoints,
            scales,
            offsets,
        ):
            # ground_truth now should already have visibility flag
            ground_truth = gt.detach().cpu().numpy()
            gt_with_vis = ground_truth

            # FIXME: convert (-1, -2) to 0. Else error is incorrectly calculated
            for batch_id in range(len(ground_truth)):
                # keypoints (num_kpts, 3)
                keypoints = ground_truth[batch_id]
                for kpts in keypoints:
                    vis = kpts[-1]
                    if vis < 0:
                        kpts[-1] = 0

            # rescale to the full image for TD or CTD
            gt_with_vis[..., :2] = (gt_with_vis[..., :2] * scale) + offset
            pred = pred.copy()
            pred[..., :2] = (pred[..., :2] * scale) + offset

            # for TD models, individuals are predicted separately
            if path in epoch_gt_metric:
                epoch_gt_metric[path] = np.concatenate(
                    [epoch_gt_metric[path], gt_with_vis], axis=0
                )
                epoch_metric[path] = np.concatenate([epoch_metric[path], pred], axis=0)
            else:
                epoch_gt_metric[path] = gt_with_vis
                epoch_metric[path] = pred

        self._epoch_ground_truth[name] = epoch_gt_metric
        self._epoch_predictions[name] = epoch_metric


class DetectorTrainingRunner(TrainingRunner[BaseDetector]):
    """Runner to train object detection models"""

    def __init__(self, model: BaseDetector, optimizer: torch.optim.Optimizer, **kwargs):
        """
        Args:
            model: The detector model to train.
            optimizer: The optimizer to use to train the model.
            **kwargs: TrainingRunner kwargs
        """
        super().__init__(model, optimizer, **kwargs)
        self._pycoco_warning_displayed = False
        self._print_valid_loss = False

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
            self.model.train()
        else:
            self.model.eval()

        images = batch["image"]
        images = images.to(self.device)

        if self.gpus:
            underlying_model = self.model.module
        else:
            underlying_model = self.model

        target = underlying_model.get_target(batch["annotations"])
        for item in target:  # target is a list here
            for key in item:
                if item[key] is not None:
                    item[key] = item[key].to(self.device)

        losses, predictions = self.model(images, target)

        # losses only returned during training, not evaluation
        if mode == "train":
            losses["total_loss"] = sum(loss_part for loss_part in losses.values())
            losses["total_loss"].backward()
            self.optimizer.step()
            losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}

        elif mode == "eval":
            losses["total_loss"] = np.nan
            self._update_epoch_predictions(
                paths=batch["path"],
                sizes=batch["original_size"],
                bboxes=batch["annotations"]["boxes"],
                predictions=predictions,
                offsets=batch["offsets"],
                scales=batch["scales"],
            )

        return losses

    def _compute_epoch_metrics(self) -> dict[str, float]:
        """Returns: bounding box metrics, if"""
        try:
            return {
                f"metrics/test.{k}": v
                for k, v in compute_bbox_metrics(
                    self._epoch_ground_truth, self._epoch_predictions
                ).items()
            }
        except ModuleNotFoundError:
            if not self._pycoco_warning_displayed:
                logging.info(
                    "\nNote:\n"
                    "Cannot compute bounding box metrics as ``pycocotools`` is not "
                    "installed. If you want bounding box mAP metrics when training "
                    "detectors for top-down models, please run ``pip install "
                    "pycocotools``.\n"
                )
                self._pycoco_warning_displayed = True

        return {}

    def _update_epoch_predictions(
        self,
        paths: torch.Tensor,
        sizes: torch.Tensor,
        bboxes: torch.Tensor,
        predictions: list[dict[str, torch.Tensor]],
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Updates the stored predictions with a new batch"""
        for img_path, img_size, img_bboxes, img_pred, scale, offset in zip(
            paths, sizes, bboxes, predictions, scales, offsets
        ):
            scale_x, scale_y = scale
            scale_factors = np.array([scale_x, scale_y, scale_x, scale_y])
            offset = np.array(offset)

            # remove bboxes that are not visible
            img_bbox_mask = (img_bboxes[:, 2] > 0.0) & (img_bboxes[:, 3] > 0.0)
            img_bboxes = img_bboxes[img_bbox_mask]

            # rescale ground truth bounding boxes
            gt_rescaled = img_bboxes.cpu().numpy() * scale_factors
            gt_rescaled[..., :2] = gt_rescaled[..., :2] + offset

            # convert to COCO format (xywh) before rescaling
            pred_rescaled = img_pred["boxes"].detach().cpu().numpy()
            pred_rescaled[:, 2] -= pred_rescaled[:, 0]
            pred_rescaled[:, 3] -= pred_rescaled[:, 1]
            pred_rescaled[..., :4] = pred_rescaled[..., :4] * scale_factors
            pred_rescaled[..., :2] = pred_rescaled[..., :2] + offset

            self._epoch_ground_truth[img_path] = {
                "bboxes": gt_rescaled,
                "width": img_size[1],
                "height": img_size[0],
            }
            self._epoch_predictions[img_path] = {
                "bboxes": pred_rescaled,
                "scores": img_pred["scores"].detach().cpu().numpy(),
            }


def build_training_runner(
    runner_config: dict,
    model_folder: Path,
    task: Task,
    model: nn.Module,
    device: str,
    gpus: list[int] | None = None,
    snapshot_path: str | None = None,
    logger: BaseLogger | None = None,
) -> TrainingRunner:
    """
    Build a runner object according to a pytorch configuration file

    Args:
        runner_config: the configuration for the runner
        model_folder: the folder where models should be saved
        task: the task the runner will perform
        model: the model to run
        device: the device to use (e.g. {'cpu', 'cuda:0', 'mps'})
        gpus: the list of GPU indices to use for multi-GPU training
        snapshot_path: the snapshot from which to load the weights
        logger: the logger to use, if any

    Returns:
        the runner that was built
    """
    optim_cfg = runner_config["optimizer"]
    optim_cls = getattr(torch.optim, optim_cfg["type"])
    optimizer = optim_cls(params=model.parameters(), **optim_cfg["params"])
    scheduler = build_scheduler(runner_config.get("scheduler"), optimizer)
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        snapshot_manager=TorchSnapshotManager(
            task=task,
            model_folder=model_folder,
            key_metric=runner_config.get("key_metric"),
            key_metric_asc=runner_config.get("key_metric_asc"),
            max_snapshots=runner_config["snapshots"]["max_snapshots"],
            save_epochs=runner_config["snapshots"]["save_epochs"],
            save_optimizer_state=runner_config["snapshots"]["save_optimizer_state"],
        ),
        device=device,
        gpus=gpus,
        eval_interval=runner_config.get("eval_interval"),
        snapshot_path=snapshot_path,
        scheduler=scheduler,
        logger=logger,
    )
    if task == Task.DETECT:
        return DetectorTrainingRunner(**kwargs)

    return PoseTrainingRunner(**kwargs)
