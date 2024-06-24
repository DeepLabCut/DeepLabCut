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

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, draw_keypoints

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

import deeplabcut.pose_estimation_pytorch.registry as deeplabcut_pose_estimation_pytorch_registry
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

LOGGER = deeplabcut_pose_estimation_pytorch_registry.Registry(
    "loggers", build_func=deeplabcut_pose_estimation_pytorch_registry.build_from_cfg
)


def setup_file_logging(filepath: Path) -> None:
    """
    Sets up logging to a file

    Args:
        filepath: the path where logs should be saved
    """
    logging.basicConfig(
        filename=filepath,
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        format="%(asctime)-15s %(message)s",
        force=True,
    )
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(console_logger)


def destroy_file_logging() -> None:
    """Resets the logging module to log everything to the console"""
    root = logging.getLogger()
    handlers = [h for h in root.handlers]
    for handler in handlers:
        root.removeHandler(handler)


class BaseLogger(ABC):
    """Base class for logging training runs"""

    @abstractmethod
    def log_config(self, config: dict = None) -> None:
        """Logs the configuration data for a training run

        Args:
            config: the training configuration used for the run
        """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Logs data from a training run

        Args:
            metrics: the metrics to log
            step: The global step in processing. Defaults to None.
        """

    @abstractmethod
    def save(self) -> None:
        """Saves the current training logs"""


class ImageLoggerMixin(ABC):
    """Mixin for loggers that can log images

    Before starting training, you should call `select_images_to_log`, which will
    select a train and a test image for which inputs/outputs will always be logged.
    Then logger.log_images should be called at every step - the logger will check if
    anything needs to be uploaded, and take care of it.

    Example:
        project_name = "example"
        run_name = "run-1"
        logger = WandbLogger(project_name, run_name)
        logger.select_images_to_log(train_loader, test_loader)

        for i in range(epochs):
            for batch_inputs in train_loader:
                batch_labels = batch_data["annotations"]
                batch_inputs = batch_data["image"]
                batch_outputs = model(batch_inputs)
                batch_targets = model.get_target(batch_outputs, batch_labels)
                loss = criterion(batch_targets, batch_outputs)
                loss.backwards()
                optim.step()

                logger.log_images(batch_inputs, batch_outputs, batch_targets)

            for batch_inputs in train_loader:
                ...
                logger.log_images(batch_inputs, batch_outputs, batch_targets)
    """

    def __init__(self, image_log_interval: int | None = None, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.image_log_interval = image_log_interval
        self._logged = {}
        self._denormalize = transforms.Compose(
            [
                transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            ]
        )
        self._softmax = torch.nn.Softmax2d()

    @abstractmethod
    def log_images(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        targets: dict[str, dict[str, torch.Tensor]],
        step: int,
    ) -> None:
        """Log images for a batch

        Args:
            inputs: the inputs for the model, containing at least an "image" key
            outputs: the outputs of each model head
            targets: the targets for each model head
            step: the current step
        """
        pass

    def select_images_to_log(self, train: DataLoader, valid: DataLoader) -> None:
        """Selects the train and test images to log

        Args:
            train: the training dataloader
            valid: the inference dataloader
        """
        def _caption(image_path: str) -> str:
            p = Path(image_path)
            return f"{p.parent.name}.{p.stem}"

        train_image = train.dataset[0]["path"]
        test_image = valid.dataset[0]["path"]
        self._logged = {
            train_image: {"name": "train-0", "caption": _caption(train_image)},
            test_image: {"name": "test-0", "caption": _caption(test_image)},
        }

    def _prepare_image(
        self,
        image: torch.Tensor,
        denormalize: bool = False,
        keypoints: torch.Tensor | None = None,
        bboxes: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Args:
            image: the image to log, of shape (C, H, W), of any data type
            denormalize: whether to remove ImageNet channel normalization
            keypoints: size (num_instances, K, 2) the K keypoints location
            bboxes: size (N, 4) containing bboxes in (xmin, ymin, xmax, ymax)

        Returns:
            an uint8 array with keypoints and bounding boxes drawn
        """
        if denormalize:
            image = self._denormalize(image.unsqueeze(0)).squeeze()

        image = F.convert_image_dtype(image.detach().cpu(), dtype=torch.uint8)
        if keypoints is not None and len(keypoints) > 0:
            assert len(keypoints.shape) == 3
            keypoints[keypoints < 0] = np.nan
            image = draw_keypoints(
                image, keypoints=keypoints[..., :2], colors="red", radius=5
            )

        if bboxes is not None and len(bboxes) > 0:
            assert len(bboxes.shape) == 2
            image = draw_bounding_boxes(image, boxes=bboxes[:, :4], width=1)

        return image.permute(1, 2, 0).numpy()

    def _heatmap_softmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Applies a softmax to the heatmap channels"""
        return self._softmax(heatmaps.detach().cpu())

    def _prepare_images(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, dict[str, torch.Tensor]]],
    ) -> dict[str, np.ndarray]:
        """Prepares images for logging"""
        image_logs = {}
        paths = inputs["path"]
        images_to_log = [(i, p) for i, p in enumerate(paths) if p in self._logged]
        for idx, path in images_to_log:
            base = self._logged[path]["name"]
            keypoints = inputs.get("annotations", {}).get("keypoints")
            if keypoints is not None:
                keypoints = keypoints[idx]
            image_logs[f"{base}.input"] = self._prepare_image(
                inputs["image"][idx], keypoints=keypoints, denormalize=True,
            )

            for head, head_outputs in outputs.items():
                if "heatmap" in head_outputs:
                    head_heatmaps = self._heatmap_softmax(head_outputs["heatmap"][idx])
                    head_targets = targets[head]["heatmap"]["target"][idx]
                    for j, (h, t) in enumerate(zip(head_heatmaps, head_targets)):
                        h = self._prepare_image(h.unsqueeze(0))
                        t = self._prepare_image(t.unsqueeze(0))
                        image_logs[f"{base}.heatmap.{j}"] = np.concatenate([h, t])

        return image_logs


@LOGGER.register_module
class WandbLogger(ImageLoggerMixin, BaseLogger):
    """Wandb logger to track experiments and log data.

    Refer to: https://docs.wandb.ai/guides for more information on wandb.

    Attributes:
        run (wandb.Run): The wandb run object associated with the current experiment.
    """

    def __init__(
        self,
        project_name: str = "deeplabcut",
        run_name: str = "tmp",
        image_log_interval: int | None = None,
        model: PoseModel = None,
        **wandb_kwargs,
    ) -> None:
        """Initialize the WandbLogger class.

        Args:
            project_name: The name of the wandb project. Defaults to "deeplabcut".
            run_name: The name of the wandb run. Defaults to "tmp".
            image_log_interval: How often train/test images are logged in epochs (if
                None, train/test inputs are never logged).
            model: The model to log. Defaults to None.
            wandb_kwargs: extra arguments to pass to ``wb.init``

        Example:
            logger = WandbLogger(project_name="mice", run_name="exp1", model=my_model)

        """
        super().__init__(image_log_interval=image_log_interval)

        if not has_wandb:
            raise ValueError(
                "Cannot use ``WandbLogger`` as wandb is not installed. Please run"
                "``pip install wandb`` if you want to log to wandb"
            )

        if wandb.run is not None:
            wandb.finish()

        self.run = wandb.init(
            project=project_name,
            name=run_name,
            **wandb_kwargs,
        )
        if model is None:
            raise ValueError("Specify the model to track!")
        self.run.watch(model)

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Logs metrics from runs

        Args:
            metrics: the metrics to log
            step: The global step in processing. Defaults to None.

        Example:
            logger = WandbLogger()
            logger.log({"loss": 0.123}, step=100)
        """
        self.run.log(metrics, step=step)

    def log_images(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, dict[str, torch.Tensor]]],
        step: int,
    ) -> None:
        """Log images for a batch

        Args:
            inputs: the inputs for the model, containing at least an "image" key
            outputs: the outputs of each model head
            targets: the targets for each model head
            step: the current step
        """
        if self.image_log_interval is None or step % self.image_log_interval != 0:
            return

        images = self._prepare_images(inputs, outputs, targets)
        if len(images) > 0:
            self.run.log(
                {name: wandb.Image(image) for name, image in images.items()},
                step=step,
            )

    def save(self):
        """Syncs all files to wandb with the policy specified.

        Notes:
            self.run: A run is a unit of computation logged by wandb.
            self.run.run.dir: The directory where files associated with the run are saved.

        Example:
            logger = WandbLogger()
            # Training and logging
            logger.save()
        """
        self.run.save(self.run.dir)

    def log_config(self, config: dict = None) -> None:
        """Updates the current run with the given config dict.

        Notes:
            self.run: A run is a unit of computation logged by wandb.
            self.run.config: Config object associated with this run.

        Args:
            config: Experiment config file.

        Example:
            logger = WandbLogger()
            config = {"learning_rate": 0.001, "batch_size": 32}
            logger.log_config(config)

        """
        self.run.config.update(config)


@LOGGER.register_module
class CSVLogger(BaseLogger):
    """Logger saving stats and metrics to a CSV file"""

    def __init__(self, train_folder: Path) -> None:
        """Initialize the WandbLogger class.

        Args:
            train_folder: The path of the folder containing training files.
        """
        super().__init__()
        self.train_folder = train_folder
        self.log_file = train_folder / "learning_stats.csv"

        self._steps: list[int] = []
        self._metric_store: list[dict] = []
        self._logged_metrics: set[str] = set()

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Logs metrics from runs

        Args:
            metrics: the metrics to log
            step: The global step in processing. Defaults to None.
        """
        if step is None:
            if len(self._steps) == 0:
                step = 0
            else:
                step = self._steps[-1] + 1

        self._logged_metrics = self._logged_metrics.union(metrics.keys())
        if len(self._steps) > 0 and step == self._steps[-1]:
            self._metric_store[-1].update(metrics)
        else:
            self._steps.append(step)
            self._metric_store.append(metrics)

        self.save()

    def save(self):
        """Saves the metrics to the file system"""
        logs = self._prepare_logs()
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(logs)

    def log_config(self, config: dict = None) -> None:
        """Does not do anything as the config should already be saved

        Args:
            config: Experiment config file.
        """
        pass

    def _prepare_logs(self) -> list[list]:
        """Prepares the data to log as a list of strings"""
        if len(self._metric_store) == 0:
            return []

        metrics = list(sorted(self._logged_metrics))
        logs = [["step"] + metrics]
        for step, step_metrics in zip(self._steps, self._metric_store):
            logs.append([step] + [step_metrics.get(m) for m in metrics])

        return logs
