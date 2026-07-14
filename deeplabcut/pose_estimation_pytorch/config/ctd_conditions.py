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

Three subclasses cover the supported input forms:

- ``ConditionsFileConfig``    — pre-computed predictions from a file (evaluation only)
- ``ConditionsModelConfig``   — fully resolved BU model (config path + snapshot path)
                                for live ``analyze_*`` inference
- ``ConditionsShuffleConfig`` — unresolved shuffle shorthand. At runtime:
                                - live inference → ``ConditionsModelConfig.resolve_from_conditions()``
                                - evaluation → ``CondFromFile(config=..., shuffle=..., ...)``

Use ``ConditionsConfig.build()`` to normalise any raw input (str, Path, dict) into one
of these types. ``build()`` is pure (no filesystem access) and safe to call from Pydantic
validators. Resolution that requires the filesystem must go through
``ConditionsModelConfig.resolve_from_conditions()``.

Context rules for ``inference.conditions`` / ``ctd_conditions``:

- **Evaluation** accepts File and Shuffle (loads pre-computed BU predictions).
- **Live analyze** (``analyze_images`` / ``analyze_videos``) accepts Shuffle and Model
  only. A predictions file path in the YAML is evaluation-only and is rejected by
  ``resolve_from_conditions``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from deeplabcut.core.config import DLCBaseConfig


class ConditionsConfig(DLCBaseConfig):
    """Base class for CTD conditions configuration.

    Use ``ConditionsConfig.build()`` to normalise any raw input into a typed subclass.

    Subclasses:
        - ``ConditionsFileConfig``    — pre-computed predictions file (evaluation only)
        - ``ConditionsModelConfig``   — resolved BU model (config + snapshot paths)
        - ``ConditionsShuffleConfig`` — unresolved shuffle shorthand (resolve to Model
          for live inference, or to ``CondFromFile`` for evaluation)
    """

    source: Literal["file", "model", "shuffle"]

    @property
    def affords_bu_inference(self) -> bool:
        """Whether this config is already a resolved BU model for live inference.

        Only ``ConditionsModelConfig`` returns ``True``. A ``ConditionsShuffleConfig``
        can be turned into one via ``ConditionsModelConfig.resolve_from_conditions()``
        but does not afford inference until resolved. ``ConditionsFileConfig`` never
        affords live BU inference (evaluation-only).
        """
        return isinstance(self, ConditionsModelConfig)

    @classmethod
    def build(
        cls,
        v: str | Path | dict | ConditionsConfig | None,
    ) -> ConditionsFileConfig | ConditionsModelConfig | ConditionsShuffleConfig | None:
        """Normalise any raw input into a typed conditions config.

        This method is pure — it never touches the filesystem. For shuffle
        shorthand inputs it returns a ``ConditionsShuffleConfig`` (unresolved).
        To obtain a fully resolved ``ConditionsModelConfig`` call
        ``ConditionsModelConfig.resolve_from_conditions()`` at the point where
        the project config is available.

        Args:
            v: Raw input. Accepted forms:
                - ``None`` or an existing ``ConditionsConfig`` → returned unchanged
                - ``str`` / ``Path`` → ``ConditionsFileConfig``
                - ``dict`` with ``filepath`` → ``ConditionsFileConfig``
                - ``dict`` with ``config_path`` + ``snapshot_path`` → ``ConditionsModelConfig``
                - ``dict`` with ``shuffle`` → ``ConditionsShuffleConfig``

        Returns:
            A typed ``ConditionsConfig`` subclass, or ``None``.
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
            if source == "shuffle" or (source is None and "shuffle" in v):
                return ConditionsShuffleConfig(**v)
            if source == "model" or "config_path" in v or "snapshot_path" in v:
                return ConditionsModelConfig(**v)
            raise ValueError(
                "Cannot determine conditions source from dict. "
                "Provide 'filepath' for a file source, "
                "'config_path'+'snapshot_path' for a model source, "
                "or 'shuffle' for a shuffle shorthand."
            )
        raise TypeError(f"Cannot build a ConditionsConfig from {type(v).__name__!r}: {v!r}")


class ConditionsFileConfig(ConditionsConfig):
    """Conditions loaded from a pre-computed predictions file (.h5, .json, .pickle).

    File-based conditions are for **evaluation only** (``load_conditions_for_evaluation``
    / ``CondFromFile``). They cannot be used for live ``analyze_images`` /
    ``analyze_videos`` inference — use a shuffle or ``ConditionsModelConfig`` instead.

    Attributes:
        filepath: Path to the predictions file.
    """

    source: Literal["file"] = "file"
    filepath: Path


class ConditionsModelConfig(ConditionsConfig):
    """Resolved config for a BU model (i.e. a snapshot ref for live inference).

    Attributes:
        config_path: Path to the BU model's ``pytorch_config.yaml``.
        snapshot_path: Path to the BU snapshot file.
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

    @classmethod
    def resolve_from_conditions(
        cls,
        conditions: dict | ConditionsShuffleConfig | ConditionsModelConfig,
        config: str | Path | None = None,
    ) -> ConditionsModelConfig:
        """Resolve conditions input to a ``ConditionsModelConfig`` for live BUCTD
        inference (``analyze_images`` / ``analyze_videos``).

        Call this in runtime code. It may touch the filesystem when resolving a
        ``ConditionsShuffleConfig`` to a ``ConditionsModelConfig``.

        Args:
            conditions: A dict, ``ConditionsShuffleConfig``, or
                ``ConditionsModelConfig``.File / path conditions cannot be resolved
                to a BU model for live inference; they are rejected.
            config: Project ``config.yaml`` path. Required when resolving a
                ``ConditionsShuffleConfig`` that does not already carry one in its
                ``config`` attribute.

        Returns:
            A resolved ``ConditionsModelConfig``.

        Raises:
            ValueError: If ``conditions`` builds to a ``ConditionsFileConfig``, or
                if a shuffle config has no project config available.
            TypeError: If ``conditions`` cannot be built into a supported type.
        """
        if not isinstance(conditions, ConditionsConfig):
            conditions = ConditionsConfig.build(conditions)

        if isinstance(conditions, ConditionsFileConfig):
            raise ValueError(
                "File-based conditions are for evaluation only and cannot be used "
                "for live BU inference. Provide a ConditionsModelConfig "
                "('config_path'+'snapshot_path') or a ConditionsShuffleConfig / "
                "shuffle dict."
            )
        if isinstance(conditions, cls):
            return conditions

        if not isinstance(conditions, ConditionsShuffleConfig):
            raise TypeError(
                f"Cannot resolve conditions of type {type(conditions).__name__} "
                "for live BU inference. Expected ConditionsShuffleConfig, "
                "ConditionsModelConfig, or a dict that builds to one of those."
            )
        cfg = conditions.config or (Path(config) if config is not None else None)
        if cfg is None:
            raise ValueError(
                "Cannot resolve shuffle conditions: no project config provided. "
                "Set 'config' in the shuffle conditions or pass it to "
                "resolve_from_conditions()."
            )
        return cls.from_shuffle(
            config=cfg,
            shuffle=conditions.shuffle,
            trainset_index=conditions.trainset_index,
            modelprefix=conditions.modelprefix,
            snapshot=conditions.snapshot,
            snapshot_index=conditions.snapshot_index,
        )


class ConditionsShuffleConfig(ConditionsConfig):
    """Unresolved shuffle shorthand for CTD conditions.

    Stores shuffle parameters without touching the filesystem. Resolve at runtime
    when the project config is available:

    - Live BU inference: ``ConditionsModelConfig.resolve_from_conditions()``
    - Evaluation predictions file: ``CondFromFile(config=..., shuffle=..., ...)``

    Attributes:
        shuffle: The index of the BU shuffle to use for conditions.
        config: Path to the DLC project ``config.yaml``. Optional — can be
            injected later via ``resolve_from_conditions(config=...)``.
        trainset_index: The TrainingsetFraction index.
        modelprefix: The model prefix for the shuffle.
        snapshot: Specific snapshot filename to use. Takes priority over
            ``snapshot_index``.
        snapshot_index: Index of the snapshot to use (default: -1, last).
    """

    source: Literal["shuffle"] = "shuffle"
    shuffle: int
    config: Path | None = None
    trainset_index: int = 0
    modelprefix: str = ""
    snapshot: str | None = None
    snapshot_index: int | None = None
