from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter
from ruamel.yaml.comments import CommentedMap
from typing_extensions import Self

from deeplabcut.core.config.mixins.mapping_access import MappingAccessMixin
from deeplabcut.core.config.utils import pretty_print, read_config_as_dict, write_config


class ConfigMixin(MappingAccessMixin):
    """Base mixin class for typed configuration dataclasses."""

    @classmethod
    def _resolve_aliases_in_dict(cls, cfg_dict: dict) -> dict:
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
        if isinstance(cfg_dict, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        cfg_dict = cls._resolve_aliases_in_dict(cfg_dict)
        TypeAdapter(cls).validate_python(cfg_dict)
        return cls(**cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Self:
        cfg_dict = cls._resolve_aliases_in_dict(cfg_dict)
        return cls(**cfg_dict)

    @classmethod
    def from_any(
        cls,
        config: Self | dict | DictConfig | str | Path,
    ) -> Self:
        if isinstance(config, ConfigMixin):
            return config
        elif isinstance(config, str | Path):
            return cls.from_yaml(config)
        elif isinstance(config, DictConfig):
            warnings.warn(
                "Passing an OmegaConf DictConfig is deprecated. Pass a plain dict or a typed config instance instead.",
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
        yaml_dict = read_config_as_dict(yaml_path)
        if ignore_empty:
            yaml_dict = {k: v for k, v in yaml_dict.items() if v is not None}
        cfg = cls.from_dict(yaml_dict)
        cfg._post_yaml_load_updates(yaml_path=Path(yaml_path))
        return cfg

    def to_yaml(
        self,
        yaml_path: str | Path,
        overwrite: bool = True,
        log_changes: bool = True,
        mark_clean: bool = True,
    ) -> None:
        dict_data = _normalize_for_serialization(self.to_dict())
        data = CommentedMap(dict_data)
        for f in fields(self):
            if comment := f.metadata.get("comment"):
                data.yaml_set_comment_before_after_key(f.name, before=comment)
        if hasattr(self, "log_changes") and log_changes:
            self.log_changes()
        if hasattr(self, "mark_clean") and mark_clean:
            self.mark_clean()
        write_config(yaml_path, data, overwrite=overwrite)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_dict_normalized(self) -> dict:
        return _normalize_for_serialization(self.to_dict())

    def to_dictconfig(self) -> DictConfig:
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

    def _post_yaml_load_updates(self, *, yaml_path: Path) -> None:
        pass


def _normalize_for_serialization(obj):
    """Recursively normalize Paths to strings and Enums to values."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Mapping):
        return type(obj)({k: _normalize_for_serialization(v) for k, v in obj.items()})
    return obj
