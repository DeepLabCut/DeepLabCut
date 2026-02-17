from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Iterator, Mapping
from typing_extensions import Self
from pathlib import Path
from dataclasses import asdict, fields
from enum import Enum
from functools import wraps

from omegaconf import OmegaConf, DictConfig, ListConfig
from pydantic import TypeAdapter
from ruamel.yaml.comments import CommentedMap

from deeplabcut.core.config.utils import read_config_as_dict, write_config, pretty_print


class ConfigMixin:
    """
    Base mixin class for configuration classes.

    Each configuration is defined as a pydantic dataclass and inherits from this
    class to provide common functionality for loading, validating, and working
    with configuration data.

    This mixin provides methods for:
    - Loading configurations from dictionaries or YAML files
    - Validating configuration data against pydantic models
    - Converting configurations to dictionaries
    - Dict-like access (cfg["key"], cfg.get("key"), "key" in cfg, etc.)
    - Pretty printing configuration data
    """

    # ------------------------------------------------------------------
    # Alias resolution
    # ------------------------------------------------------------------

    @classmethod
    @functools.cache
    def _alias_map(cls) -> dict[str, str]:
        """Build ``{alias: canonical_name}`` from field metadata.

        Each field may declare ``metadata={"aliases": ["old_name", ...]}``.
        The mapping is computed once per class and cached.

        Raises:
            ValueError: if two fields declare the same alias.
        """
        mapping: dict[str, str] = {}
        for f in fields(cls):
            for alias in f.metadata.get("aliases", []):
                if alias in mapping:
                    raise ValueError(
                        f"Duplicate alias '{alias}' for fields "
                        f"'{mapping[alias]}' and '{f.name}'"
                    )
                mapping[alias] = f.name
        return mapping

    def _resolve_alias(self, name: str) -> str | None:
        """Return the canonical field name if *name* is a known alias, else ``None``."""
        return type(self)._alias_map().get(name)

    def _warn_alias(self, alias: str, canonical: str, stacklevel: int = 3) -> None:
        """Emit a deprecation warning for alias usage."""
        warnings.warn(
            f"'{alias}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    # ------------------------------------------------------------------
    # Dict-like access protocol (with alias support)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails (i.e. for aliases).
        canonical = self._resolve_alias(name)
        if canonical is not None:
            self._warn_alias(name, canonical, stacklevel=2)
            return getattr(self, canonical)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def __getitem__(self, key: str) -> Any:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            return getattr(self, canonical)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            key = canonical
        if key not in self._field_names():
            raise KeyError(
                f"'{type(self).__name__}' has no field '{key}'"
            )
        object.__setattr__(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._field_names():
            return True
        return self._resolve_alias(key) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names())

    def __len__(self) -> int:
        return len(fields(self))

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible .get() with an optional default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return field names (like dict.keys())."""
        return self._field_names()

    def values(self) -> list[Any]:
        """Return field values (like dict.values())."""
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        """Return (name, value) pairs (like dict.items())."""
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def select(self, path: str, default: Any = None) -> Any:
        """Nested dot-path access into this config.

        Replacement for ``OmegaConf.select(cfg, "a.b.c")``.

        Args:
            path: Dot-separated key path (e.g. ``"data.train.top_down_crop"``).
            default: Value to return when a segment is missing.

        Returns:
            The value at the given path, or *default* if any segment is missing.
        """
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
        return [f.name for f in fields(self)]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_aliases_in_dict(cls, cfg_dict: dict) -> dict:
        """Rename alias keys to their canonical names in a config dict.

        This is called before passing the dict to the Pydantic constructor,
        so that old YAML files / dicts with deprecated key names load
        correctly (and a :class:`DeprecationWarning` is emitted for each
        alias key found).
        """
        alias_map = cls._alias_map()
        if not alias_map:
            return cfg_dict

        resolved = {}
        for k, v in cfg_dict.items():
            canonical = alias_map.get(k)
            if canonical is not None:
                warnings.warn(
                    f"Config key '{k}' is deprecated, use '{canonical}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                k = canonical
            resolved[k] = v
        return resolved

    @classmethod
    def validate_dict(cls, cfg_dict: dict | DictConfig) -> Self:
        """Validate a dictionary against this config's pydantic model.

        Args:
            cfg_dict: the configuration as a dictionary (or DictConfig during
                the deprecation transition).

        Returns:
            A validated instance of this configuration class.
        """
        if isinstance(cfg_dict, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        cfg_dict = cls._resolve_aliases_in_dict(cfg_dict)
        TypeAdapter(cls).validate_python(cfg_dict)
        return cls(**cfg_dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Self:
        cfg_dict = cls._resolve_aliases_in_dict(cfg_dict)
        return cls(**cfg_dict)

    @classmethod
    def from_any(
        cls,
        config: Self | dict | DictConfig | str | Path,
    ) -> Self:
        """
        Create a new instance from various configuration formats.

        Args:
            config: Configuration as a ConfigMixin instance, dictionary,
                    DictConfig (deprecated), string, or Path.
                    If already a ConfigMixin instance, returns it as-is.

        Returns:
            A new instance of the ConfigMixin subclass.
        """
        if isinstance(config, ConfigMixin):
            return config
        elif isinstance(config, str | Path):
            return cls.from_yaml(config)
        elif isinstance(config, DictConfig):
            warnings.warn(
                "Passing an OmegaConf DictConfig is deprecated. "
                "Pass a plain dict or a typed config instance instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls.from_dict(OmegaConf.to_container(config, resolve=True))
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise TypeError(
                "Failure to load configuration: Expected pydantic dataclass, "
                f"dictionary, DictConfig, string, or Path. Got {type(config)}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, ignore_empty: bool = True) -> Self:
        """
        Load a configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            ignore_empty: If True, empty/None values in the YAML are ignored and
                dataclass defaults are used instead. Defaults to True.

        Returns:
            A new instance of the configuration class.
        """
        # NOTE @deruyter92 2026-02-05: Default ignore_empty is now set to True to match
        # the prior behaviour of read_config. We should consider changing this to False
        # for stricter validation.
        yaml_dict = read_config_as_dict(yaml_path)
        if ignore_empty:
            yaml_dict = {k: v for k, v in yaml_dict.items() if v is not None}
        return cls.from_dict(yaml_dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_yaml(self, yaml_path: str | Path, overwrite: bool = True) -> None:
        dict_data = self.to_dict_normalized()
        data = CommentedMap(dict_data)
        for f in fields(self):
            if (comment := f.metadata.get("comment")):
                data.yaml_set_comment_before_after_key(f.name, before=comment)
        write_config(yaml_path, data, overwrite=overwrite)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_dict_normalized(self) -> dict:
        return _normalize_for_serialization(self.to_dict())

    def to_dictconfig(self) -> DictConfig:
        """Convert to an OmegaConf DictConfig.

        .. deprecated::
            This method will be removed in a future version.
            Use the typed config instance directly (it supports dict-like access)
            or call ``.to_dict()`` if a plain dict is needed.
        """
        warnings.warn(
            "to_dictconfig() is deprecated. Use the typed config instance directly "
            "(it supports dict-like access) or call .to_dict() for a plain dict.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OmegaConf.create(self.to_dict())

    def print(
        self,
        indent: int = 0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        pretty_print(config=self.to_dict(), indent=indent, print_fn=print_fn)


def _to_plain(value):
    """Convert a ConfigMixin, DictConfig, or ListConfig to a plain Python object."""
    if isinstance(value, ConfigMixin):
        return value.to_dict()
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def ensure_plain_config(fn: Callable) -> Callable:
    """Decorator that converts config arguments to plain Python dicts.

    Any positional or keyword argument that is a :class:`ConfigMixin`,
    :class:`~omegaconf.DictConfig`, or :class:`~omegaconf.ListConfig` is
    automatically converted to a plain ``dict`` / ``list`` before the
    decorated function is called.

    Example::

        @ensure_plain_config
        def train(model_cfg: dict, lr: float = 1e-3):
            ...

        train(my_pose_config)  # PoseConfig → dict
        train(omega_dict)      # DictConfig → dict
        train(plain_dict)      # dict passed through unchanged
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(_to_plain(a) for a in args)
        kwargs = {k: _to_plain(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


def _normalize_for_serialization(obj: Any) -> Any:
    """Recursively normalize Paths to strings and Enums to values."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Mapping):
        return type(obj)({k: _normalize_for_serialization(v) for k, v in obj.items()})
    return obj