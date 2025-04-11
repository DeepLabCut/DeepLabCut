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

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.data.snapshots import list_snapshots, Snapshot


@dataclass
class TorchSnapshotManager:
    """Class handling model checkpoint I/O

    Attributes:
        snapshot_prefix: The prefix to use when saving snapshots.
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
            "snapshot",
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

    snapshot_prefix: str
    model_folder: Path
    key_metric: str | None = None
    key_metric_asc: bool = True
    max_snapshots: int = 5
    save_epochs: int = 25
    save_optimizer_state: bool = False
    _best_model_epochs: int = -1
    _best_metric: float | None = None
    _key: str = field(init=False)

    def __post_init__(self):
        assert self.max_snapshots > 0, f"max_snapshots must be a positive integer"
        self._key = f"metrics/{self.key_metric}"

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
            self._key in metrics
            and not np.isnan(metrics[self._key])
            and (
                self._best_metric is None
                or (self.key_metric_asc and self._best_metric < metrics[self._key])
                or (not self.key_metric_asc and self._best_metric > metrics[self._key])
            )
        ):
            current_best = self.best()
            self._best_metric = metrics[self._key]

            # Save the new best model
            save_path = self.snapshot_path(epoch, best=True)
            parsed_state_dict = {
                k: v
                for k, v in state_dict.items()
                if self.save_optimizer_state or k != "optimizer"
            }
            torch.save(parsed_state_dict, save_path)

            # Handle previous best model
            if current_best is not None:
                if current_best.epochs % self.save_epochs == 0:
                    new_name = self.snapshot_path(epoch=current_best.epochs)
                    current_best.path.rename(new_name)
                else:
                    current_best.path.unlink(missing_ok=False)
        elif last or epoch % self.save_epochs == 0:
            # Save regular snapshot if needed
            save_path = self.snapshot_path(epoch=epoch)
            parsed_state_dict = {
                k: v
                for k, v in state_dict.items()
                if self.save_optimizer_state or k != "optimizer"
            }
            torch.save(parsed_state_dict, save_path)

        # Clean up old snapshots if needed
        existing_snapshots = [s for s in self.snapshots() if not s.best]
        if len(existing_snapshots) >= self.max_snapshots:
            num_to_delete = len(existing_snapshots) - self.max_snapshots
            to_delete = existing_snapshots[:num_to_delete]
            for snapshot in to_delete:
                snapshot.path.unlink(missing_ok=False)

    def best(self) -> Snapshot | None:
        """Returns: the path to the best snapshot, if it exists"""
        snapshots = self.snapshots()
        best_snapshots = [s for s in snapshots if s.best]
        if len(best_snapshots) == 0:
            return None

        if len(best_snapshots) > 1:
            warnings.warn(
                f"TorchSnapshotManager.best(): found multiple best snapshots ("
                f"{best_snapshots}), returning the last one."
            )

        best_snapshot = best_snapshots[-1]
        return best_snapshot

    def last(self) -> Snapshot | None:
        """Returns: path to the last snapshot that was saved, if any snapshot exists"""
        snapshots = self.snapshots(best_in_last=False)
        if len(snapshots) == 0:
            return None
        return snapshots[-1]

    def snapshots(self, best_in_last: bool = True) -> list[Snapshot]:
        """
        Args:
            best_in_last: Whether to place the snapshot with the best performance in the
                last position in the list, even if it wasn't the last epoch.

        Returns:
            The snapshots for a training run, sorted by the number of epochs they were
            trained for. If ``best_in_last=True`` and a best snapshot exists, it will be
            the last one in the list.
        """
        return list_snapshots(
            self.model_folder, self.snapshot_prefix, best_in_last=best_in_last
        )

    def snapshot_path(self, epoch: int, best: bool = False) -> Path:
        """
        Args:
            epoch: the number of epochs for which a snapshot was trained
            best: whether this is the best performing model for the training run

        Returns:
            the path where the model should be stored
        """
        uid = f"{epoch:03}"
        if best:
            uid = f"best-{uid}"
        return self.model_folder / f"{self.snapshot_prefix}-{uid}.pt"
