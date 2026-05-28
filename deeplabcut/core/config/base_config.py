from __future__ import annotations

import functools
import logging
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_core import ArgsKwargs
from ruamel.yaml.comments import CommentedMap
from typing_extensions import Self

from deeplabcut.core.config import versioning
from deeplabcut.core.config.utils import (
    normalize_for_serialization,
    pretty_print,
    read_config_as_dict,
    resolve_aliases_in_dict,
    write_config,
)
from deeplabcut.core.config.versioning import migrate_config

logger = logging.getLogger(__name__)


class DLCBaseConfig(BaseModel):
    """Pydantic base for DeepLabCut configuration models.

    This class is used to create configuration models for DeepLabCut.
    It provides a base class for all configuration models that need YAML/dict I/O
    and optional deprecated field names via ``json_schema_extra["aliases"]``.
    (Use for all nested configs, e.g. pytorch ``DataConfig``, ``InferenceConfig``, etc.)

    For project-level schema migration and dirty-field tracking, subclass
    `DLCVersionedConfig` instead.

    Features:

    - Strict schema (`extra="forbid"`, `validate_assignment=True`).
    - Load and save: `from_yaml`, `from_dict`, `from_any`, `to_yaml`, `to_dict`.
    - Pretty-print via `print`.
    - Hooks: `_post_yaml_load_updates`.
    - Nested dot-notation via `select`.
    - Dict-like access over declared fields (legacy compatibility).
    - Field aliases from `json_schema_extra`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ------------------------------------------------------------------
    # Validation (before pydantic field validation)
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases_before_validate(cls, data: Any) -> Any:
        """Resolves aliases to their canonical names. (Normalizes ArgsKwargs
        input to a dict for downstream validation.)

        Args:
            data: Raw validator input (`dict`, `ArgsKwargs`, or other).

        Returns:
            A dict with canonical field names when input is ArgsKwargs or dict;
            otherwise `data` unchanged.
        """
        if isinstance(data, ArgsKwargs):
            data: dict = cls._args_kwargs_to_dict(data)
        if isinstance(data, dict):
            return resolve_aliases_in_dict(data, cls._alias_map(), target=cls.__name__)
        return data

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Self:
        return cls.model_validate(cfg_dict)

    @classmethod
    def from_any(
        cls,
        config: Self | dict | str | Path,
    ) -> Self:
        if isinstance(config, cls):
            return config
        elif isinstance(config, str | Path):
            return cls.from_yaml(config)
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise TypeError(
                "Failure to load configuration: Expected a config instance, "
                f"dictionary, string, or Path. Got {type(config)}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, ignore_empty: bool = True) -> Self:
        yaml_dict = read_config_as_dict(yaml_path)
        if ignore_empty:
            yaml_dict = {k: v for k, v in yaml_dict.items() if v is not None}
        cfg = cls.from_dict(yaml_dict)
        cfg._post_yaml_load_updates(yaml_path=Path(yaml_path))
        return cfg

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_yaml(
        self,
        yaml_path: str | Path,
        *,
        overwrite: bool = True,
    ) -> None:
        dict_data = self.to_dict(normalize=True)
        data = CommentedMap(dict_data)
        for name, info in type(self).model_fields.items():
            extra = info.json_schema_extra
            if isinstance(extra, dict) and (comment := extra.get("comment")):
                data.yaml_set_comment_before_after_key(name, before=comment)
        write_config(yaml_path, data, overwrite=overwrite)

    def to_dict(self, *, normalize: bool = False) -> dict:
        if not normalize:
            return self.model_dump()
        return normalize_for_serialization(self.model_dump())

    def print(
        self,
        indent: int = 0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        pretty_print(config=self.to_dict(), indent=indent, print_fn=print_fn)

    # ------------------------------------------------------------------
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------

    def _post_yaml_load_updates(self, *, yaml_path: Path) -> None:
        pass

    # ------------------------------------------------------------------
    # Field aliases (deprecated names in json_schema_extra)
    # ------------------------------------------------------------------

    @classmethod
    @functools.cache
    def _alias_map(cls) -> dict[str, str]:
        """Build a map of deprecated aliases to canonical field names.

        Returns:
            Dict mapping each alias in `json_schema_extra["aliases"]` to its
            canonical field name.

        Raises:
            ValueError: If the same alias is declared on more than one field.
        """
        mapping: dict[str, str] = {}
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            if not isinstance(extra, dict):
                continue
            for alias in extra.get("aliases", []):
                if alias in mapping:
                    raise ValueError(f"Duplicate alias '{alias}' for fields '{mapping[alias]}' and '{name}'")
                mapping[alias] = name
        return mapping

    def _resolve_alias(
        self,
        name: str,
        *,
        warn: bool = False,
        stacklevel: int = 3,
    ) -> str:
        canonical = type(self)._alias_map().get(name)
        if canonical is not None:
            if warn:
                from deeplabcut.utils.deprecation import DLCDeprecationWarning

                warnings.warn(
                    f"'{name}' is deprecated, use '{canonical}' instead.",
                    DLCDeprecationWarning,
                    stacklevel=stacklevel,
                )
            return canonical
        return name

    # ------------------------------------------------------------------
    # Dict-like access (canonical field names only in keys()/iter)
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        name = self._resolve_alias(name, warn=True, stacklevel=3)
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name not in type(self)._alias_map():
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self, self._resolve_alias(name, warn=True, stacklevel=2))

    def __getitem__(self, key: str) -> Any:
        key = self._resolve_alias(key, warn=True, stacklevel=3)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        canonical = self._resolve_alias(key, warn=False)
        if canonical not in self._field_names():
            raise KeyError(f"'{type(self).__name__}' has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._field_names():
            return True
        return key in type(self)._alias_map()

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names())

    def __len__(self) -> int:
        return len(self._field_names())

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return self._field_names()

    def values(self) -> list[Any]:
        return [getattr(self, name) for name in self._field_names()]

    def items(self) -> list[tuple[str, Any]]:
        return [(name, getattr(self, name)) for name in self._field_names()]

    def select(self, path: str, default: Any = None) -> Any:
        obj: Any = self
        for part in path.split("."):
            if obj is None:
                return default
            try:
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            except (KeyError, AttributeError, TypeError):
                return default
        return obj

    def _field_names(self) -> list[str]:
        cls = type(self)
        if not isinstance(self, BaseModel):
            raise TypeError(f"{cls.__name__} must inherit from pydantic.BaseModel")
        return list(cls.model_fields.keys())

    @classmethod
    def _args_kwargs_to_dict(cls, data: ArgsKwargs) -> dict:
        """Map positional and keyword constructor args to a field-name dict."""
        names = list(cls.model_fields.keys())
        return dict(
            zip(names, data.args or [], strict=False),
            **(data.kwargs or {}),
        )


class DLCVersionedConfig(DLCBaseConfig):
    """Top-level configs with schema migration and change tracking.

    Subclass of `DLCBaseConfig` for project and pose YAML configs such as
    `ProjectConfig` and `PoseConfig`.

    Note:
        Pydantic runs `migrate_before_validate` before the base
        `resolve_aliases_before_validate` (child-first order): schema migration
        on legacy keys, then alias resolution for the current model.

    Additional behavior:

    - `migrate_before_validate` upgrades raw dicts to `CURRENT_CONFIG_VERSION`.
    - Tracks fields modified after load; `to_yaml` can log changes and mark clean.
    - Patches `__setattr__` once per class to record dirty fields while delegating
      alias warnings to the base `__setattr__`.
    """

    _CHANGE_TRACKING_INTERNALS: ClassVar[frozenset[str]] = frozenset(
        {
            "_dirty_fields",
            "_change_notes",
            "_change_tracking_initialized",
        }
    )

    # ------------------------------------------------------------------
    # Version migration (before pydantic field validation)
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def migrate_before_validate(cls, data: Any) -> Any:
        """Upgrade a raw config dict to `CURRENT_CONFIG_VERSION`.

        Args:
            data: Raw validator input (`ArgsKwargs` or `dict`).

        Returns:
            Migrated dict for input dict or ArgsKwargs. Unchanged otherwise.
        """
        if isinstance(data, ArgsKwargs):
            data = cls._args_kwargs_to_dict(data)
        if isinstance(data, dict):
            data = migrate_config(data, target_version=versioning.CURRENT_CONFIG_VERSION)
        return data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_yaml(
        self,
        yaml_path: str | Path,
        *,
        overwrite: bool = True,
        log_changes: bool = True,
        mark_clean: bool = True,
    ) -> None:
        super().to_yaml(yaml_path, overwrite=overwrite)
        if log_changes:
            self.log_changes()
        if mark_clean:
            self.mark_clean()

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._init_change_tracking()

    def _init_change_tracking(self) -> None:
        if getattr(self, "_change_tracking_initialized", False):
            return
        object.__setattr__(self, "_change_tracking_initialized", True)

        cls = type(self)
        if not getattr(cls, "_change_tracking_installed", False):
            original_setattr = cls.__setattr__

            def __setattr__(self, name: str, value: Any) -> None:
                if name in type(self)._CHANGE_TRACKING_INTERNALS:
                    object.__setattr__(self, name, value)
                    return
                canonical = self._resolve_alias(name, warn=False)
                field_names = list(type(self).model_fields.keys())
                dirty_fields = getattr(self, "_dirty_fields", None)
                if dirty_fields is not None and canonical in field_names:
                    old = getattr(self, canonical)
                    original_setattr(self, name, value)
                    if old != value:
                        dirty_fields.add(canonical)
                else:
                    original_setattr(self, name, value)

            cls.__setattr__ = __setattr__
            cls._change_tracking_installed = True

        object.__setattr__(self, "_dirty_fields", set())
        object.__setattr__(self, "_change_notes", {})

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty_fields)

    @property
    def dirty_fields(self) -> frozenset[str]:
        return frozenset(self._dirty_fields)

    @property
    def change_notes(self) -> list[str]:
        return list(self._change_notes.values())

    def record_change_note(
        self,
        field_name: str,
        message: str,
        *,
        include_caller: bool = False,
        _stack_depth: int = 1,
    ) -> None:
        if include_caller:
            frame = sys._getframe(_stack_depth)
            filename = frame.f_code.co_filename.rsplit("/", 1)[-1]
            message = f"{message} [{filename}:{frame.f_lineno}]"
        self._change_notes[field_name] = message

    def log_changes(self) -> None:
        if not self.is_dirty:
            return
        logger.info(f"Updates to {type(self).__name__}:")
        for field_name in sorted(self._dirty_fields):
            if field_name in self._change_notes:
                logger.info(f"  {self._change_notes[field_name]}")
            else:
                logger.info(f"  {field_name} was modified")

    def mark_clean(self) -> None:
        self._dirty_fields.clear()
        self._change_notes.clear()
