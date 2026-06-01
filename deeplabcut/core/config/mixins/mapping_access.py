from __future__ import annotations

import functools
import warnings
from collections.abc import Iterator
from dataclasses import fields
from typing import Any


class MappingAccessMixin:
    """Dict-like access protocol for typed config dataclasses.

    Supports key access, membership checks, iteration helpers, and dot-path
    selection while preserving dataclass field semantics.
    """

    @classmethod
    @functools.cache
    def _alias_map(cls) -> dict[str, str]:
        """Build ``{alias: canonical_name}`` from field metadata."""
        mapping: dict[str, str] = {}
        for f in fields(cls):
            for alias in f.metadata.get("aliases", []):
                if alias in mapping:
                    raise ValueError(f"Duplicate alias '{alias}' for fields '{mapping[alias]}' and '{f.name}'")
                mapping[alias] = f.name
        return mapping

    def _resolve_alias(self, name: str) -> str | None:
        return type(self)._alias_map().get(name)

    def _warn_alias(self, alias: str, canonical: str, stacklevel: int = 3) -> None:
        warnings.warn(
            f"'{alias}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    def __getattr__(self, name: str) -> Any:
        canonical = self._resolve_alias(name)
        if canonical is not None:
            self._warn_alias(name, canonical, stacklevel=2)
            return getattr(self, canonical)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            return getattr(self, canonical)
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value: Any) -> None:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            key = canonical
        if key not in self._field_names():
            raise KeyError(f"'{type(self).__name__}' has no field '{key}'")
        setattr(self, key, value)

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
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return self._field_names()

    def values(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

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
        return [f.name for f in fields(self)]
