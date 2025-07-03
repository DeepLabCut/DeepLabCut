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
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Snapshot:
    """A snapshot for a model"""

    best: bool
    epochs: int | None
    path: Path

    def uid(self) -> str:
        if self.best:
            return f"best-{self.epochs}"
        else:
            return str(self.epochs)

    @staticmethod
    def from_path(path: Path) -> "Snapshot":
        best = "-best" in path.stem
        # Use regex to extract epoch number more robustly
        match = re.search(r'-(\d+)\.pt$', path.name)
        if match:
            epochs = int(match.group(1))
        else:
            # Fallback to original method if regex fails
            epochs = int(path.stem.split("-")[-1])
        return Snapshot(best=best, epochs=epochs, path=path)


def list_snapshots(
    model_folder: Path,
    snapshot_prefix: str,
    best_in_last: bool = True,
) -> list[Snapshot]:
    """Lists snapshots in a model folder.

    Args:
        model_folder: The model in which the snapshots are found.
        snapshot_prefix: The prefix for the snapshot names.
        best_in_last: Whether to place the snapshot with the best performance in the
            last position in the list, even if it wasn't the last epoch.

    Returns:
        The snapshots stored in a folder, sorted by the number of epochs they were
        trained for. If ``best_in_last=True`` and a best snapshot exists, it will be
        the last one in the list.
    """
    def _sort_key(snapshot: Snapshot) -> int:
        return snapshot.epochs

    def _sort_key_best_as_last(snapshot: Snapshot) -> tuple[int, int]:
        return 1 if snapshot.best else 0, snapshot.epochs

    pattern = r"^(" + snapshot_prefix + r"(-best)?-\d+\.pt)$"
    snapshots = [
        Snapshot.from_path(f) for f in model_folder.iterdir() if re.match(pattern, f.name)
    ]

    sort_key = _sort_key
    if best_in_last:
        sort_key = _sort_key_best_as_last
    snapshots.sort(key=sort_key)
    return snapshots
