"""Opt-in dirty-state tracking mixin for pydantic dataclass configs."""
from __future__ import annotations

import logging
import sys
from typing import Any

from dataclasses import fields

logger = logging.getLogger(__name__)


class ChangeTrackingMixin:
    """Opt-in mixin that records which fields are *dirty* (modified but not
    yet written to disk).

    Tracking is installed lazily on first instantiation so it wraps previously
    set up ``__setattr__`` (incl. pydantic's ``validate_assignment``).

    A field is marked dirty when the validated write succeeds and the new
    value differs from the old one.

    Human-readable *change notes* can be attached to dirty fields via
    :meth:`record_change_note`.  These are *not* generated automatically.

    Tracking state is stored in ``__slots__`` so it survives pydantic's
    ``__dict__`` rebuilds during validated assignment.

    Example::

        @dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
        class MyConfig(ChangeTrackingMixin, ConfigMixin):
            ...
    """

    __slots__ = ("_dirty_fields", "_change_notes")

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        cls = type(self)
        if not getattr(cls, "_change_tracking_installed", False):
            original_setattr = cls.__setattr__

            def __setattr__(self, name: str, value: Any) -> None:
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
        """True if any field has been modified since construction or the last
        call to :meth:`mark_clean`."""
        return bool(self._dirty_fields)

    @property
    def dirty_fields(self) -> frozenset[str]:
        """Field names modified since construction or the last call to
        :meth:`mark_clean`."""
        return frozenset(self._dirty_fields)

    @property
    def change_notes(self) -> list[str]:
        """Human-readable notes describing dirty-field changes."""
        return list(self._change_notes.values())

    def record_change_note(
        self,
        field_name: str,
        message: str,
        *,
        include_caller: bool = False,
        _stack_depth: int = 1,
    ) -> None:
        """Attach a human-readable note to a dirty field.

        Overwrites any previous note for the same *field_name*.

        Args:
            include_caller: If True, appends a ``[file:line]`` caller tag.
            _stack_depth: Frames to skip when resolving the caller (increase
                when wrapping this method).
        """
        if include_caller:
            frame = sys._getframe(_stack_depth)
            filename = frame.f_code.co_filename.rsplit("/", 1)[-1]
            message = f"{message} [{filename}:{frame.f_lineno}]"
        self._change_notes[field_name] = message

    def log_changes(self) -> None:
        """Log all dirty fields at INFO level.

        Fields with an explicit change note show the note; the rest are
        listed by name only.
        """
        if not self.is_dirty:
            return
        logger.info(f"Updates to {type(self).__name__}:")
        for field_name in sorted(self._dirty_fields):
            if field_name in self._change_notes:
                logger.info(f"  {self._change_notes[field_name]}")
            else:
                logger.info(f"  {field_name} was modified")

    def mark_clean(self) -> None:
        """Reset dirty tracking, treating the current state as clean."""
        self._dirty_fields.clear()
        self._change_notes.clear()
