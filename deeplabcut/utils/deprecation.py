#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

P = ParamSpec("P")
R = TypeVar("R")


class DLCDeprecationWarning(DeprecationWarning):
    """Project-specific deprecation warning. Helps with filtering."""


class DeprecationInfo(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    kind: Literal["callable", "parameter"]
    target: str
    replacement: str | None = None

    since: Version | None = None
    removed_in: Version | None = None

    old_parameter: str | None = None
    new_parameter: str | None = None

    @field_validator("since", "removed_in", mode="before")
    @classmethod
    def _parse_version(cls, value):
        if value is None or isinstance(value, Version):
            return value
        try:
            return Version(value)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version: {value!r}") from e

    @model_validator(mode="after")
    def _validate_version_order(self) -> DeprecationInfo:
        if self.since and self.removed_in and self.removed_in <= self.since:
            raise ValueError(f"'removed_in' ({self.removed_in}) must be greater than 'since' ({self.since}).")
        return self

    def format_message(self) -> str:
        if self.kind == "callable":
            parts = [f"{self.target} is deprecated"]
            if self.since:
                parts[0] += f" since {self.since}"
            if self.replacement:
                parts.append(f"Use {self.replacement} instead.")
            if self.removed_in:
                parts.append(f"It will be removed in {self.removed_in}.")
            return " ".join(parts)

        if self.kind == "parameter":
            return (
                f"Parameter '{self.old_parameter}' of {self.target} is deprecated"
                + (f" since {self.since}" if self.since else "")
                + f"; use '{self.new_parameter}' instead."
            )

        raise ValueError(f"Unknown deprecation kind: {self.kind}")


def deprecated(
    *,
    replacement: str | None = None,
    since: str | None = None,
    removed_in: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as deprecated.

    Args:
        replacement: Fully-qualified name of the replacement callable, e.g.
            ``"deeplabcut.utils.auxfun_videos.list_videos_in_folder"``.
        since: Version in which the function was deprecated.
        removed_in: Version in which the function will be removed.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        info = DeprecationInfo(
            kind="callable",
            target=fn.__qualname__,
            replacement=replacement,
            since=since,
            removed_in=removed_in,
        )
        message = info.format_message()

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(message, DLCDeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__doc__ = f"Deprecated. {message}\n\n" + (fn.__doc__ or "")
        wrapper.__deprecated_info__ = info
        return wrapper

    return decorator


def renamed_parameter(
    *,
    old: str,
    new: str,
    since: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Support a renamed keyword argument while warning callers to update.

    Args:
        old: The old parameter name that callers may still pass.
        new: The current parameter name the function actually accepts.
        since: Version when the rename happened.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        info = DeprecationInfo(
            kind="parameter",
            target=fn.__qualname__,
            since=since,
            old_parameter=old,
            new_parameter=new,
        )
        message = info.format_message()

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{fn.__qualname__} received both '{old}' and '{new}'. Use only '{new}'.")
                warnings.warn(message, DLCDeprecationWarning, stacklevel=2)
                kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        existing = getattr(fn, "__deprecated_params__", ())
        wrapper.__deprecated_params__ = (*existing, info)
        return wrapper

    return decorator
