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


def deprecated(
    replacement: str | None = None,
    since: str | None = None,
    removed_in: str | None = None,
) -> Callable:
    """Mark a function as deprecated.

    Args:
        replacement: Fully-qualified name of the replacement callable, e.g.
            ``"deeplabcut.utils.auxfun_videos.list_videos_in_folder"``.
        since: Version in which the function was deprecated.
        removed_in: Version in which the function will be removed.
    """

    def decorator(fn: Callable) -> Callable:
        parts = [f"{fn.__qualname__} is deprecated"]
        if since:
            parts[0] += f" since {since}"
        if replacement:
            parts.append(f"Use {replacement} instead.")
        if removed_in:
            parts.append(f"It will be removed in {removed_in}.")
        message = " ".join(parts)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__doc__ = f"Deprecated. {message}\n\n" + (fn.__doc__ or "")
        return wrapper

    return decorator


def renamed_parameter(old: str, new: str, since: str | None = None) -> Callable:
    """Support a renamed keyword argument while warning callers to update.

    Args:
        old: The old parameter name that callers may still pass.
        new: The current parameter name the function actually accepts.
        since: Version when the rename happened.
    """

    def decorator(fn: Callable) -> Callable:
        msg = (
            f"Parameter '{old}' of {fn.__qualname__} is deprecated"
            + (f" since {since}" if since else "")
            + f"; use '{new}' instead."
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if old in kwargs:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
