"""Opt-in dirty-state tracking mixin for pydantic dataclass configs."""

from __future__ import annotations

import logging
import sys
from dataclasses import fields
from typing import Any

logger = logging.getLogger(__name__)


class ChangeTrackingMixin:
    """Opt-in mixin that records which fields are dirty."""

    __slots__ = ("_dirty_fields", "_change_notes")

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        cls = type(self)
        if not getattr(cls, "_change_tracking_installed", False):
            original_setattr = cls.__setattr__

            def __setattr__(self, name: str, value: Any) -> None:
                if name in ("_dirty_fields", "_change_notes"):
                    object.__setattr__(self, name, value)
                    return
                field_names = [f.name for f in fields(self)]
                dirty_fields = getattr(self, "_dirty_fields", None)
                if dirty_fields is not None and name in field_names:
                    old = getattr(self, name)
                    original_setattr(self, name, value)
                    if old != value:
                        dirty_fields.add(name)
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
