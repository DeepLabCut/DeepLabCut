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
Two subclasses are supported (``ConditionsFileConfig`` and ``ConditionsModelConfig``)
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

    @property
    def affords_bu_inference(self) -> bool:
        """Whether the config affords BU inference (requires a BU model)."""
        return isinstance(self, ConditionsModelConfig)

    def assert_bu_inference(self) -> None:
        """Raise a ValueError if not configured for BU inference."""
        if not self.affords_bu_inference:
            raise ValueError(
                "This operation requires a BU model to be configured as conditions "
                f"(ConditionsModelConfig), but got {type(self).__name__}. "
                "Provide 'config_path' and 'snapshot_path', or use the shuffle shorthand."
            )

    @classmethod
    def build(
        cls,
        v: str | Path | dict | ConditionsConfig | None,
    ) -> ConditionsFileConfig | ConditionsModelConfig | None:
        """Build a typed conditions config from any supported input.

        Returns:
            ConditionsFileConfig | ConditionsModelConfig | None
            - If None or already-typed instance, returned unchanged.
            - If ``str`` / ``Path`` or ``dict`` with ``source="file"``, returns a
              ``ConditionsFileConfig``.
            - If ``dict`` with ``source="model"`` and ``config_path``/``snapshot_path``,
              returns a ``ConditionsModelConfig`` directly.
            - If ``dict`` with ``source="model"`` and ``config``/``shuffle`` keys
              (shuffle shorthand), resolves the shuffle via ``ConditionsModelConfig.from_shuffle``
              (touches the filesystem) and returns a ``ConditionsModelConfig``.
        """
        if v is None or isinstance(v, ConditionsConfig):
            return v
        if isinstance(v, (str, Path)):
            return ConditionsFileConfig(filepath=Path(v))
        if isinstance(v, dict):
            v = v.copy()
            source = v.pop("source", None)
            if source == "file" or (source is None and "filepath" in v):
                return ConditionsFileConfig(**v)
            if "config" in v and "shuffle" in v:
                return ConditionsModelConfig.from_shuffle(**v)
            if source == "model" or "config_path" in v or "snapshot_path" in v:
                return ConditionsModelConfig(**v)
            raise ValueError(
                "Cannot determine conditions source from dict. "
                "Provide 'filepath' for a file source, "
                "'config_path'+'snapshot_path' for a model source, "
                "or 'config'+'shuffle' for a shuffle shorthand."
            )
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

    Attributes:
        config_path: Path to the BU model's ``pytorch_config.yaml`` (direct form).
        snapshot_path: Path to the BU snapshot file (direct form).
        scorer: Scorer name for the BU model. Used to look for pre-computed
            conditions files on disk before running the model.
    """

    source: Literal["model"] = "model"
    config_path: Path
    snapshot_path: Path
    scorer: str | None = None

    @classmethod
    def from_shuffle(
        cls,
        config: str | Path,
        shuffle: int,
        trainset_index: int = 0,
        modelprefix: str = "",
        snapshot: str | None = None,
        snapshot_index: int | None = None,
    ) -> ConditionsModelConfig:
        """Resolve a DLC BU shuffle to its model config and snapshot paths."""
        from deeplabcut.pose_estimation_pytorch.data.ctd import resolve_bu_shuffle

        loader, bu_snapshot = resolve_bu_shuffle(config, shuffle, trainset_index, modelprefix, snapshot, snapshot_index)
        return cls(
            config_path=loader.model_config_path,
            snapshot_path=bu_snapshot.path,
            scorer=loader.scorer(bu_snapshot),
        )
