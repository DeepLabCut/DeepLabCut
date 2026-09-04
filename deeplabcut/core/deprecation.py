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
import inspect
import warnings
from collections.abc import Callable
from enum import Enum, unique
from typing import Literal, ParamSpec, TypeVar

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

P = ParamSpec("P")
R = TypeVar("R")


class DLCDeprecationWarning(DeprecationWarning):
    """Project-specific deprecation warning. Helps with filtering."""


class DeprecationRoundInfo(BaseModel):
    """Description of a round of deprecations introduced together."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    since: Version
    removed_in: Version | None = None
    pull_request: int | None = None
    summary: str | None = None

    @field_validator("since", "removed_in", mode="before")
    @classmethod
    def _coerce_version(cls, value):
        if value is None or isinstance(value, Version):
            return value
        try:
            return Version(value)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version: {value!r}") from e

    @model_validator(mode="after")
    def _validate_version_order(self) -> Self:
        if self.removed_in and self.removed_in <= self.since:
            raise ValueError(f"'removed_in' ({self.removed_in}) must be greater than 'since' ({self.since}).")
        return self

    @property
    def url(self) -> str | None:
        if self.pull_request is None:
            return None
        return f"https://github.com/DeepLabCut/DeepLabCut/pull/{self.pull_request}"


@unique
class DeprecationRound(Enum):
    """Collection of all deprecation rounds used in the deprecation markers - single source of truth."""

    INIT_PARAMETER_ALIASING = DeprecationRoundInfo(
        since="3.0.0",
        pull_request=3332,
        summary=(
            "Apply consistent naming in parameters (spacing, casing, etc). "
            "e.g. renamed videotype -> video_extensions, batchsize -> batch_size, "
            "Shuffles -> shuffles"
        ),
    )

    VIDEO_PATH_MIGRATION = DeprecationRoundInfo(
        since="3.0.0",
        pull_request=3303,
        summary="Migrate get_list_of_videos and friends to deeplabcut.collect_video_paths",
    )

    PATHLIB_MIGRATION = DeprecationRoundInfo(
        since="3.0.1",
        pull_request=3421,
        summary="Migrate str path handling to pathlib.Path, including grab_files_in_folder",
    )

    CONFIG_MODEL_MIGRATION = DeprecationRoundInfo(
        since="3.0.1",
        pull_request=3198,
        summary="Migrate raw config dicts and their helpers to validated config models",
    )

    INIT_TF_DEPRECATION = DeprecationRoundInfo(
        since="3.1",
        removed_in="3.2",
        pull_request=3382,
        summary=(
            "Deprecate the TensorFlow-only API in preparation for dropping TensorFlow. "
            "e.g. create_pretrained_human_project -> create_pretrained_project(model='full_human'), "
            "analyze_time_lapse_frames -> analyze_images"
        ),
    )


class DeprecationInfo(BaseModel):
    """Description for a single deprecation marker."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    kind: Literal["callable", "parameter"]
    target: str
    replacement: str | None = None
    deprecation_round: DeprecationRound | None = None

    old_parameter: str | None = None
    new_parameter: str | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> Self:
        if self.kind == "callable":
            if self.old_parameter is not None or self.new_parameter is not None:
                raise ValueError("Callable deprecations cannot specify parameter names.")
        elif self.kind == "parameter":
            if not self.old_parameter or not self.new_parameter:
                raise ValueError("Parameter deprecations require both 'old_parameter' and 'new_parameter'.")
            if self.replacement is not None:
                raise ValueError("Parameter deprecations cannot specify 'replacement'.")
        return self

    @property
    def since(self) -> Version | None:
        return self.deprecation_round.value.since if self.deprecation_round else None

    @property
    def removed_in(self) -> Version | None:
        return self.deprecation_round.value.removed_in if self.deprecation_round else None

    @property
    def url(self) -> str | None:
        return self.deprecation_round.value.url if self.deprecation_round else None

    def format_message(self) -> str:
        if self.kind == "callable":
            parts = [f"{self.target} is deprecated"]
            if self.since:
                parts[0] += f" since {self.since}"
            if self.replacement:
                parts.append(f"Use {self.replacement} instead.")
            if self.removed_in:
                parts.append(f"It will be removed in {self.removed_in}.")
            if self.url:
                parts.append(f"See {self.url} for more details.")
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
    deprecation_round: DeprecationRound | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as deprecated.

    Args:
        replacement: Fully-qualified name of the replacement callable, e.g.
            ``"deeplabcut.utils.auxfun_videos.list_videos_in_folder"``.
        deprecation_round: A ``DeprecationRound`` member (add a new one there if needed).
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        info = DeprecationInfo(
            kind="callable",
            target=fn.__qualname__,
            replacement=replacement,
            deprecation_round=deprecation_round,
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
    deprecation_round: DeprecationRound | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Support a renamed keyword argument while warning callers to update.

    Args:
        old: The old parameter name that callers may still pass.
        new: The current parameter name the function actually accepts.
        deprecation_round: A ``DeprecationRound`` member (add a new one there if needed).

    Rules:
        - ``new`` must be the name used in the function signature and all
          internal call-sites.  ``old`` must **not** appear in the signature.
        - Do **not** chain renames.  If ``A`` was renamed to ``B`` and ``B``
          is later renamed to ``C``, replace the ``A→B`` decorator with
          ``A→C`` directly rather than stacking a second decorator.
            Example:
                @renamed_parameter(old="A", new="C", deprecation_round=DeprecationRound.FIRST_RENAME)
                @renamed_parameter(old="B", new="C", deprecation_round=DeprecationRound.SECOND_RENAME)
                def func(*, C: int):
                    print(f"C={C}")
        - Multiple independent renames on the same function (e.g.
          ``batchsize→batch_size`` *and* ``videotype→video_extensions``) are fine
          as long as they do not form a chain.
        - This decorator only intercepts **keyword** arguments.  Positional
          arguments are passed through unchanged; renaming a parameter that
          callers commonly pass positionally will not be caught.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(fn)

        # Guard: disallow chaining renames (A→B stacked on top of B→C).
        existing = getattr(fn, "__deprecated_params__", ())
        for prev in existing:
            if prev.old_parameter == new:
                raise ValueError(
                    f"@renamed_parameter: chaining renames is not allowed. "
                    f"'{old}' → '{new}' would chain with the existing "
                    f"'{prev.old_parameter}' → '{prev.new_parameter}' rename "
                    f"on {fn.__qualname__}. "
                    f"Use '{old}' → '{prev.new_parameter}' directly instead."
                )

        # Guard: 'new' must actually exist in the function's signature.
        if new not in sig.parameters:
            raise ValueError(
                f"@renamed_parameter: '{new}' is not a parameter of "
                f"{fn.__qualname__}. "
                f"Available parameters: {list(sig.parameters)}"
            )

        # Guard: 'old' must NOT exist in the signature.
        if old in sig.parameters:
            raise ValueError(
                f"@renamed_parameter: '{old}' is still a parameter of "
                f"{fn.__qualname__}. Use either old name or new name: '{new}'."
            )

        info = DeprecationInfo(
            kind="parameter",
            target=fn.__qualname__,
            deprecation_round=deprecation_round,
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

        wrapper.__deprecated_params__ = (*existing, info)
        return wrapper

    return decorator
