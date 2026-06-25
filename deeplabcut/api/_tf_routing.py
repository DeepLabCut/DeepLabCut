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
"""
Routing for legacy TensorFlow API while still supported. Remove this module when TF support is dropped.
"""

import warnings
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine
from deeplabcut.utils.auxiliaryfunctions import read_config
from deeplabcut.utils.deprecation import DLCDeprecationWarning

_TF_MODULE = "deeplabcut.tensorflow_compat.tensorflow_api"

_DROPPED_TF_KWARGS = frozenset(
    {
        "allow_growth",
        "autotune",
        "superanimal_name",
        "superanimal_transfer_learning",
        "save_iters",
        "max_iters",
    }
)

_RENAMED_TF_KWARGS: dict[str, tuple[str, Callable]] = {
    "gputouse": ("device", lambda v: f"cuda:{v}" if isinstance(v, int) else v),
}


@lru_cache
def _get_tensorflow_impl(name: str):
    return getattr(import_module(_TF_MODULE), name)


def warn_deprecated_tensorflow():
    warnings.warn(
        "\n"
        "━" * 60 + "\n"
        "⚠️  DeepLabCut — TensorFlow support is deprecated\n"
        "━" * 60 + "\n"
        "TensorFlow support will be removed in a future release.\n"
        "Your project config and annotated data are fully compatible with PyTorch.\n"
        "Please run create_training_dataset with any PyTorch model architecture to switch to PyTorch.\n"
        "See our docs for more information: https://deeplabcut.github.io/DeepLabCut/docs/pytorch/architectures.html\n"
        "━" * 60,
        DLCDeprecationWarning,
        stacklevel=3,
    )


def with_tensorflow_fallback(_fn: Callable | None = None, *, tensorflow_name: str | None = None) -> Callable:
    """Decorator for wrapping canonical PyTorch API functions, routing to a fallback TF function if required.
    It automatically resolves the engine and converts legacy TensorFlow kwargs to canonical PyTorch kwargs, if needed.
    Can be used with or without parentheses.

    Args:
        tensorflow_name: The name of the fallback TensorFlow function in ``_TF_MODULE``. If not specified, uses the name
            of the canonical PyTorch function.

    Note:
        The engine is resolved from the shuffle metadata if not specified explicitly. If neither ``shuffles``,
        ``shuffle`` or ``engine`` is passed, it assumes shuffle=1.
    """

    def decorator(fn):
        tf_name = tensorflow_name or fn.__name__

        def wrapper(*args, **kwargs):
            engine = _resolve_engine(*args, **kwargs)
            kwargs.pop("engine", None)
            if engine == Engine.TF:
                warn_deprecated_tensorflow()
                return _get_tensorflow_impl(tf_name)(*args, **kwargs)
            kwargs = _resolve_legacy_kwargs(kwargs)
            return fn(*args, **kwargs)

        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator


def _resolve_engine(*args, **kwargs) -> Engine:
    """Resolve engine from explicit engine parameter or shuffle metadata."""
    engine = kwargs.get("engine")
    if engine is not None:
        return engine

    # Resolve engine from shuffle (default to shuffle=1 if not specified)
    shuffles = kwargs.get("shuffles") or [kwargs.get("shuffle", 1)]
    cfg = read_config(kwargs.get("config", args[0]))
    engines = {
        get_shuffle_engine(
            cfg,
            trainingsetindex=kwargs.get("trainingsetindex", 0),
            shuffle=s,
            modelprefix=kwargs.get("modelprefix", ""),
        )
        for s in shuffles
    }
    if len(engines) > 1:
        raise ValueError(f"All shuffles must have the same engine (found different engines for shuffles: {shuffles}).")
    return engines.pop()


def _resolve_legacy_kwargs(kwargs: dict) -> dict:
    """Resolve legacy TensorFlow kwargs to canonical (PyTorch) kwargs."""
    kwargs = dict(kwargs)
    for old, (new, convert) in _RENAMED_TF_KWARGS.items():
        if old in kwargs and new in kwargs:
            raise TypeError(f"Cannot specify both '{old}' (deprecated) and '{new}'. Use '{new}' only.")
        elif old in kwargs:
            converted = convert(kwargs.pop(old))
            kwargs[new] = converted
            warnings.warn(
                f"'{old}' is deprecated; use {new}='{converted}' instead.",
                DLCDeprecationWarning,
                stacklevel=3,
            )
    for key in _DROPPED_TF_KWARGS:
        if key in kwargs:
            kwargs.pop(key)
            warnings.warn(
                f"'{key}' is a TensorFlow-only parameter and has no effect for PyTorch projects.",
                DLCDeprecationWarning,
                stacklevel=3,
            )
    return kwargs
