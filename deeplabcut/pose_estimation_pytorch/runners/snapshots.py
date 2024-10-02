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
"""Code to handle storing models"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.task import Task


@dataclass(frozen=True)
class Snapshot:
    best: bool
    epochs: int | None
    path: Path

    def uid(self) -> str:
        return self.path.stem.split("-")[-1]


@dataclass
class TorchSnapshotManager:
    """Class handling model checkpoint I/O

    Attributes:
        task: The task that the model is performing.
        model_folder: The path to the directory where model snapshots should be stored.
        key_metric: If defined, the metric is used to save the best model. Otherwise no
            best model is used.
        key_metric_asc: Whether the key metric is ascending (larger values are better).
        max_snapshots: The maximum number of snapshots to store for the training run.
            This does not include the best model (e.g., setting max_snapshots=5 will
            mean that the 5 latest models will be kept, plus the best model)
        save_epochs: The number of epochs between each model save
        save_optimizer_state: Whether to store the optimizer state. This makes snapshots
            much heavier, but allows to resume training as if it was never stopped.

    Examples:
        # Storing snapshots while training
        model: nn.Module
        loader = DLCLoader(...)
        snapshot_manager = TorchSnapshotManager(
            Task.BOTTOM_UP,
            loader.model_folder,
            key_metric="test.mAP",
        )
        ...
        for epoch in range(num_epochs):
            train_epoch(model, data)
            snapshot_manager.update({
                "metadata": {
                    "metrics": {"mAP": ...}
                },
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            })
    """
    task: Task
    model_folder: Path
    key_metric: str | None = None
    key_metric_asc: bool = True
    max_snapshots: int = 5
    save_epochs: int = 25
    save_optimizer_state: bool = False

    def __post_init__(self):
        assert self.max_snapshots > 0, f"max_snapshots must be a positive integer"
        self._best_model_epochs = -1
        self._best_metric = None

    def update(self, epoch: int, state_dict: dict, last: bool = False) -> None:
        """Saves the model state dict if the epoch is one that requires a save

        Args:
            epoch: the number of epochs the model was trained for
            state_dict: the state dict to store
            last: whether this is the last epoch in the training run, which forces a
                model save no matter the epoch number

        Returns:
            the path to the saved snapshot if one
        """
        metrics = state_dict["metadata"]["metrics"]
        if (
            self.key_metric in metrics and
            not np.isnan(metrics[self.key_metric]) and (
                self._best_metric is None or
                (self.key_metric_asc and self._best_metric < metrics[self.key_metric]) or
                (not self.key_metric_asc and self._best_metric > metrics[self.key_metric])
            )
        ):
            print(f"Saving best snapshot at epoch={epoch}")
            self._best_metric = metrics[self.key_metric]
            save_path = self.snapshot_path(best=True)
            parsed_state_dict = {
                k: v
                for k, v in state_dict.items()
                if self.save_optimizer_state or k != "optimizer"
            }
            torch.save(parsed_state_dict, save_path)

        if not (last or epoch % self.save_epochs == 0):
            return

        existing_snapshots = self.snapshots(include_best=False)
        if len(existing_snapshots) >= self.max_snapshots:
            num_to_delete = 1 + len(existing_snapshots) - self.max_snapshots
            to_delete = existing_snapshots[:num_to_delete]
            for snapshot in to_delete:
                snapshot.path.unlink(missing_ok=False)

        save_path = self.snapshot_path(epoch=epoch)
        parsed_state_dict = {
            k: v
            for k, v in state_dict.items()
            if self.save_optimizer_state or k != "optimizer"
        }
        torch.save(parsed_state_dict, save_path)

    def best(self) -> Snapshot | None:
        """Returns: the path to the best snapshot, if it exists"""
        best_path = self.snapshot_path(best=True)
        if not best_path.exists():
            return None
        return Snapshot(best=True, epochs=None, path=best_path)

    def snapshots(self, include_best: bool = True) -> list[Snapshot]:
        """
        Args:
            include_best: whether to return the path to the best snapshot as well

        Returns:
            The paths to snapshots for a training run, sorted by the number of epochs
            they were trained for. If the best_snapshot is returned, it's the last one
            in the list.
        """
        pattern = r"^(" + self.task.snapshot_prefix + r"-\d+\.pt)$"
        snapshots = [
            Snapshot(best=False, epochs=int(f.stem.split("-")[-1]), path=f)
            for f in self.model_folder.iterdir() if re.match(pattern, f.name)
        ]
        snapshots.sort(key=lambda s: s.epochs)

        if include_best and (best_snapshot := self.best()) is not None:
            snapshots.append(best_snapshot)

        return snapshots

    def snapshot_path(self, epoch: int | None = None, best: bool = False) -> Path:
        """
        Args:
            epoch: the number of epochs for which a snapshot was trained
            best: whether this is the best performing model for the training run

        Returns:
            the path where the model should be stored
        """
        if epoch is None and not best:
            raise ValueError(f"For non-best models, the epochs must be specified")
        uid = "best" if best else f"{epoch:03}"
        return self.model_folder / f"{self.task.snapshot_prefix}-{uid}.pt"
