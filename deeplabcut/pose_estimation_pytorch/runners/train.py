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
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

import deeplabcut.core.metrics as metrics
import deeplabcut.pose_estimation_pytorch.runners.schedulers as schedulers
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
from deeplabcut.pose_estimation_pytorch.runners.base import (
    attempt_snapshot_load,
    ModelType,
    Runner,
)
from deeplabcut.pose_estimation_pytorch.runners.logger import (
    BaseLogger,
    CSVLogger,
    ImageLoggerMixin,
)
from deeplabcut.pose_estimation_pytorch.runners.snapshots import TorchSnapshotManager
from deeplabcut.pose_estimation_pytorch.task import Task


class TrainingRunner(Runner, Generic[ModelType], metaclass=ABCMeta):
    """Base TrainingRunner class.

    A TrainingRunner is used to fit models to datasets. Subclasses must implement the
    ``step(self, batch, mode)`` method, which performs a single training or validation
    step on a batch of data. The step is different depending on the model type (e.g.
    a pose model step vs. an object detector step).

    Args:
        model: The model to fit.
        optimizer: The optimizer to use to fit the model.
        snapshot_manager: Manages how snapshots are saved to disk during training.
        device: The device on which to run training (e.g. 'cpu', 'cuda', 'cuda:0').
        gpus: Used to specify the GPU indices for multi-GPU training (e.g. [0, 1, 2, 3]
            to train on 4 GPUs). When a GPUs list is given, the device must be 'cuda'.
        eval_interval: The interval at which the model will be evaluated while training
            (e.g. `eval_interva=5` means the model will be evaluated every 5 epochs).
        snapshot_path: If continuing to train a model, the path to the snapshot to
            resume training from.
        scheduler: The learning rate scheduler (or it's configuration), if one should be
            used.
        load_scheduler_state_dict: When resuming training (snapshot_path is not None),
            attempts to load the scheduler state dict from the snapshot. If you've
            modified your scheduler, set this to False or the old scheduler parameters
            might be used.
        logger: Logger to monitor training (e.g. a WandBLogger).
        log_filename: Name of the file in which to store training stats.
        load_weights_only: Value for the torch.load() `weights_only` parameter if
            `snapshot_path` is not None.
            If False, the python pickle module is used implicitly, which is known to
            be insecure. Only set to False if you're loading data that you trust
            (e.g. snapshots that you created yourself). For more information, see:
                https://pytorch.org/docs/stable/generated/torch.load.html
            If None, the default value is used:
                `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`
    """

    def __init__(
        self,
        model: ModelType,
        optimizer: dict | torch.optim.Optimizer,
        snapshot_manager: TorchSnapshotManager,
        device: str = "cpu",
        gpus: list[int] | None = None,
        eval_interval: int = 1,
        snapshot_path: str | Path | None = None,
        scheduler: dict | torch.optim.lr_scheduler.LRScheduler | None = None,
        load_scheduler_state_dict: bool = True,
        logger: BaseLogger | None = None,
        log_filename: str = "learning_stats.csv",
        load_weights_only: bool | None = None,
    ):
        super().__init__(
            model=model, device=device, gpus=gpus, snapshot_path=snapshot_path
        )
        if isinstance(optimizer, dict):
            optimizer = build_optimizer(model, optimizer)
        if isinstance(scheduler, dict):
            scheduler = schedulers.build_scheduler(scheduler, optimizer)

        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.snapshot_manager = snapshot_manager
        self.history: dict[str, list] = dict(train_loss=[], eval_loss=[])
        self.csv_logger = CSVLogger(
            train_folder=snapshot_manager.model_folder,
            log_filename=log_filename,
        )
        self.logger = logger
        self.starting_epoch = 0
        self.current_epoch = 0

        # some models cannot compute a validation loss (e.g. detectors)
        self._print_valid_loss = True

        if self.snapshot_path:
            snapshot = self.load_snapshot(
                self.snapshot_path,
                self.device,
                self.model,
                weights_only=load_weights_only,
            )
            self.starting_epoch = snapshot.get("metadata", {}).get("epoch", 0)

            if "optimizer" in snapshot:
                self.optimizer.load_state_dict(snapshot["optimizer"])

            self._load_scheduler_state_dict(load_scheduler_state_dict, snapshot)

        self._metadata = dict(epoch=self.starting_epoch, metrics=dict(), losses=dict())
        self._epoch_ground_truth = {}
        self._epoch_predictions = {}

    def state_dict(self) -> dict:
        """Returns: the state dict for the runner"""
        model = self.model
        if self._data_parallel:
            model = self.model.module

        state_dict_ = dict(
            metadata=self._metadata,
            model=model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.scheduler is not None:
            state_dict_["scheduler"] = self.scheduler.state_dict()

        return state_dict_

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
        if self._data_parallel:
            self.model = DataParallel(self.model, device_ids=self._gpus).cuda()
        else:
            self.model.to(self.device)

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.select_images_to_log(train_loader, valid_loader)

        # continuing to train a model: either total epochs or extra epochs
        if self.starting_epoch > 0:
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

            epoch_metrics = self._metadata.get("metrics")
            if (
                e % self.eval_interval == 0
                and epoch_metrics is not None
                and len(epoch_metrics) > 0
            ):
                logging.info(f"Model performance:")
                line_length = max([len(name) for name in epoch_metrics.keys()]) + 2
                for name, score in epoch_metrics.items():
                    logging.info(f"  {(name + ':').ljust(line_length)}{score:6.2f}")

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
            if "total_loss" in losses_dict:
                epoch_loss.append(losses_dict["total_loss"])
                if (i + 1) % display_iters == 0 and mode != "eval":
                    logging.info(
                        f"Number of iterations: {i + 1}, "
                        f"loss: {losses_dict['total_loss']:.5f}, "
                        f"lr: {self.optimizer.param_groups[0]['lr']}"
                    )

            for key in losses_dict.keys():
                loss_metrics[key].append(losses_dict[key])

        perf_metrics = None
        if mode == "eval":
            perf_metrics = self._compute_epoch_metrics()
            self._metadata["metrics"] = perf_metrics
            self._epoch_predictions = {}
            self._epoch_ground_truth = {}

        if len(epoch_loss) > 0:
            epoch_loss = np.mean(epoch_loss).item()
        else:
            epoch_loss = 0
        self.history[f"{mode}_loss"].append(epoch_loss)

        metrics_to_log = {}
        if perf_metrics:
            for name, score in perf_metrics.items():
                if not isinstance(score, (int, float)):
                    score = 0.0
                metrics_to_log[name] = score

        for key in loss_metrics:
            name = f"{mode}.{key}"
            val = float("nan")
            if np.sum(~np.isnan(loss_metrics[key])) > 0:
                val = np.nanmean(loss_metrics[key]).item()
            self._metadata["losses"][name] = val
            metrics_to_log[f"losses/{name}"] = val

        self.csv_logger.log(metrics_to_log, step=self.current_epoch)
        if self.logger:
            self.logger.log(metrics_to_log, step=self.current_epoch)

        return epoch_loss

    def _load_scheduler_state_dict(self, load_state_dict: bool, snapshot: dict) -> None:
        if self.scheduler is None:
            return

        loaded_state_dict = False
        if load_state_dict and "scheduler" in snapshot:
            try:
                schedulers.load_scheduler_state(self.scheduler, snapshot["scheduler"])
                loaded_state_dict = True
            except ValueError as err:
                logging.warning(
                    "Failed to load the scheduler state_dict. The scheduler will "
                    "restart at epoch 0. This is expected if the scheduler "
                    "configuration was edited since the original snapshot was "
                    f"trained. Error: {err}"
                )

        if not loaded_state_dict and self.starting_epoch > 0:
            logging.info(
                f"Setting the scheduler starting epoch to {self.starting_epoch}"
            )
            self.scheduler.last_epoch = self.starting_epoch


class PoseTrainingRunner(TrainingRunner[PoseModel]):
    """Runner to train pose estimation models"""

    def __init__(
        self,
        model: PoseModel,
        optimizer: torch.optim.Optimizer,
        load_head_weights: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: The neural network for solving pose estimation task.
            optimizer: A PyTorch optimizer for updating model parameters.
            load_head_weights: When `snapshot_path` is not None, whether to load the
                head weights from the saved snapshot or just the backbone weights.
            **kwargs: TrainingRunner kwargs
        """
        self._load_head_weights = load_head_weights
        super().__init__(model, optimizer, **kwargs)

    def load_snapshot(
        self,
        snapshot_path: str | Path,
        device: str,
        model: PoseModel,
        weights_only: bool | None = None,
    ) -> dict:
        """Loads the state dict for a model from a file

        This method loads a file containing a DeepLabCut PyTorch model snapshot onto
        a given device, and sets the model weights using the state_dict.

        Args:
            snapshot_path: the path containing the model weights to load
            device: the device on which the model should be loaded
            model: the model for which the weights are loaded
            weights_only: Value for torch.load() `weights_only` parameter.
                If False, the python pickle module is used implicitly, which is known to
                be insecure. Only set to False if you're loading data that you trust
                (e.g. snapshots that you created yourself). For more information, see:
                    https://pytorch.org/docs/stable/generated/torch.load.html
                If None, the default value is used:
                    `deeplabcut.pose_estimation_pytorch.get_load_weights_only()`

        Returns:
            The content of the snapshot file.
        """
        snapshot = attempt_snapshot_load(snapshot_path, device, weights_only)
        if self._load_head_weights:
            model.load_state_dict(snapshot["model"])
        else:
            backbone_prefix = "backbone."
            backbone_weights = {
                k[len(backbone_prefix) :]: v
                for k, v in snapshot["model"].items()
                if k.startswith(backbone_prefix)
            }
            model.backbone.load_state_dict(backbone_weights)

        return snapshot

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
        if 'cond_keypoints' in batch['context']:
            cond_kpts = batch['context']['cond_keypoints']
            outputs = self.model(inputs, cond_kpts=cond_kpts)
        else:
            outputs = self.model(inputs)

        if self._data_parallel:
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
                gt_keypoints=ground_truth,
                pred_keypoints=predictions["bodypart"]["poses"],
                offsets=batch["offsets"],
                scales=batch["scales"],
            )
            if "unique_bodypart" in predictions:
                self._update_epoch_predictions(
                    name="unique_bodyparts",
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
        scores = metrics.compute_metrics(
            ground_truth=self._epoch_ground_truth["bodyparts"],
            predictions=self._epoch_predictions["bodyparts"],
            single_animal=False,
            unique_bodypart_gt=self._epoch_ground_truth.get("unique_bodyparts"),
            unique_bodypart_poses=self._epoch_predictions.get("unique_bodyparts"),
            pcutoff=0.6,
            compute_detection_rmse=False,
        )
        return {f"metrics/test.{metric}": value for metric, value in scores.items()}

    def _update_epoch_predictions(
        self,
        name: str,
        gt_keypoints: torch.Tensor,
        pred_keypoints: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Updates the stored predictions with a new batch"""
        epoch_gt_metric = self._epoch_ground_truth.get(name, {})
        epoch_metric = self._epoch_predictions.get(name, {})
        assert len(gt_keypoints) == len(pred_keypoints)
        assert len(offsets) == len(scales)
        scales = scales.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()

        for gt, pred, scale, offset in zip(
            gt_keypoints,
            pred_keypoints,
            scales,
            offsets,
        ):
            ground_truth = gt.detach().cpu().numpy()
            pred = pred.copy()

            # rescale to the full image for TD or CTD
            ground_truth[..., :2] = (ground_truth[..., :2] * scale) + offset
            pred[..., :2] = (pred[..., :2] * scale) + offset

            # we don't care about image paths here - use a default index
            index = len(epoch_metric) + 1
            epoch_gt_metric[f"sample{index:09}"] = ground_truth
            epoch_metric[f"sample{index:09}"] = pred

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
        log_filename = "learning_stats_detector.csv"
        if "log_filename" in kwargs:
            log_filename = kwargs.pop("log_filename")

        super().__init__(model, optimizer, log_filename=log_filename, **kwargs)
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

        if self._data_parallel:
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
            losses["total_loss"] = float("nan")
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
                for k, v in metrics.compute_bbox_metrics(
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
    snapshot_path: str | Path | None = None,
    load_head_weights: bool = True,
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
        load_head_weights: When `snapshot_path` is not None and a pose model is being
            trained, whether to load the head weights from the saved snapshot.
        logger: the logger to use, if any

    Returns:
        the runner that was built
    """
    optimizer = build_optimizer(model, runner_config["optimizer"])
    scheduler = schedulers.build_scheduler(runner_config.get("scheduler"), optimizer)

    # if no custom snapshot prefix is defined, use the default one
    snapshot_prefix = runner_config.get("snapshot_prefix")
    if snapshot_prefix is None or len(snapshot_prefix) == 0:
        snapshot_prefix = task.snapshot_prefix

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        snapshot_manager=TorchSnapshotManager(
            snapshot_prefix=snapshot_prefix,
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
        load_scheduler_state_dict=runner_config.get("load_scheduler_state_dict", True),
        logger=logger,
        load_weights_only=runner_config.get("load_weights_only", None),
    )
    if task == Task.DETECT:
        return DetectorTrainingRunner(**kwargs)

    kwargs["load_head_weights"] = load_head_weights
    return PoseTrainingRunner(**kwargs)


def build_optimizer(
    model: nn.Module,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """Builds an optimizer from a configuration.

    Args:
        model: The model to optimize.
        optimizer_config: The configuration for the optimizer.

    Returns:
        The optimizer for the model built according to the given configuration.
    """
    optim_cls = getattr(torch.optim, optimizer_config["type"])
    optimizer = optim_cls(params=model.parameters(), **optimizer_config["params"])
    return optimizer
