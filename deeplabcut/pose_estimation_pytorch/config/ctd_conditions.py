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
"""Typed configuration for CTD (Conditional Top-Down) model conditions.
Two subclasses are supported (``ConditionsFileConfig`` and `ConditionsModelConfig``)
depending on the source of the conditions (a file path, or a reference to a BU
model). Use ``ConditionsConfig.build()`` to construct from any raw input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from deeplabcut.core.config import DLCBaseConfig


class ConditionsConfig(DLCBaseConfig):
    """Base class for CTD conditions configuration.

    Use ``ConditionsConfig.build()`` to construct from any supported input,
    or instantiate a subclass directly:

    - ``ConditionsFileConfig`` load pre-computed predictions from a file
    - ``ConditionsModelConfig`` run a BU model snapshot live at inference time
    """

    source: Literal["file", "model"]

    @classmethod
    def build(
        cls,
        v: str | Path | dict | ConditionsConfig | None,
    ) -> ConditionsFileConfig | ConditionsModelConfig | None:
        """Build a typed conditions config from any supported input.

        Returns:
            ConditionsFileConfig | ConditionsModelConfig | None
            - If None or already-typed instance, returned unchanged.
            - If ``str`` / ``Path`` or ``dict`` with ``source="file"``, returns a ``ConditionsFileConfig``
            - If ``dict`` with ``source="model"``, returns a ``ConditionsModelConfig``
        """
        if v is None or isinstance(v, ConditionsConfig):
            return v
        if isinstance(v, (str, Path)):
            return ConditionsFileConfig(filepath=Path(v))
        if isinstance(v, dict):
            source = v.get("source")
            if source == "file" or (source is None and "filepath" in v):
                return ConditionsFileConfig.model_validate(v if source else {"source": "file", **v})
            return ConditionsModelConfig.model_validate(v if source else {"source": "model", **v})
        raise TypeError(f"Cannot build a ConditionsConfig from {type(v).__name__!r}: {v!r}")


class ConditionsFileConfig(ConditionsConfig):
    """Conditions loaded from a pre-computed predictions file (.h5, .json, .pickle).

    Attributes:
        filepath: Path to the predictions file.
    """

    source: Literal["file"] = "file"
    filepath: Path


class ConditionsModelConfig(ConditionsConfig):
    """Conditions generated at inference time by running a BU model snapshot.

    Two construction modes:

    Directly provide the BU model config and snapshot explicitly:

        config_path:   /path/to/model-dir/pytorch_config.yaml
        snapshot_path: /path/to/model-dir/snapshot-best.pth

    DLC shuffle shorthand to resolve from a project shuffle:

        config:         /path/to/project/config.yaml
        shuffle:        1
        snapshot:       snapshot-250.pt   # or use snapshot_index

    Attributes:
        config_path: Path to the BU model's ``pytorch_config.yaml`` (direct form).
        snapshot_path: Path to the BU snapshot file (direct form).
        scorer: Scorer name for the BU model. Used to look for pre-computed
            conditions files on disk before running the model.
        config: Path to the DLC project ``config.yaml`` (shuffle shorthand).
        shuffle: Shuffle index (shuffle shorthand).
        snapshot: Snapshot filename within the shuffle (shuffle shorthand).
        snapshot_index: Snapshot index within the shuffle (shuffle shorthand).
        trainset_index: Training-set fraction index (shuffle shorthand).
        modelprefix: Model prefix for the shuffle (shuffle shorthand).
    """

    source: Literal["model"] = "model"
    # Direct form
    config_path: Path | None = None
    snapshot_path: Path | None = None
    scorer: str | None = None
    # Shuffle shorthand
    config: Path | None = None
    shuffle: int | None = None
    snapshot: str | None = None
    snapshot_index: int | None = None
    trainset_index: int = 0
    modelprefix: str = ""
