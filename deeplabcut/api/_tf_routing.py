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
from pathlib import Path

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
        "See our docs for more information: https://deeplabcut.github.io/DeepLabCut/docs/pytorch/architectures.html \n"
        "━" * 60,
        DLCDeprecationWarning,
        stacklevel=3,
    )


def with_tensorflow_fallback(canonical_function=None, *, tensorflow_name=None):
    """Use as @with_tensorflow_fallback or @with_tensorflow_fallback()."""

    def decorator(fn):
        tf_name = tensorflow_name or fn.__name__

        def wrapper(*args, **kwargs):
            engine = resolve_engine(
                config=kwargs.get("config", args[0]),
                shuffle=kwargs.get("shuffle", 1),
                trainingsetindex=kwargs.get("trainingsetindex", 0),
                modelprefix=kwargs.get("modelprefix", ""),
                engine=kwargs.get("engine"),
            )
            kwargs.pop("engine", None)
            if engine == Engine.TF:
                warn_deprecated_tensorflow()
                return _get_tensorflow_impl(tf_name)(*args, **kwargs)
            kwargs = _resolve_legacy_kwargs(kwargs)
            return fn(*args, **kwargs)

        return wrapper

    if canonical_function is not None:
        return decorator(canonical_function)
    return decorator


def resolve_engine(
    *,
    config: str | Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    modelprefix: str = "",
    engine: Engine | None = None,
) -> Engine:
    """Resolve engine from explicit override or shuffle metadata."""
    if engine is not None:
        return engine
    return get_shuffle_engine(
        read_config(config),
        trainingsetindex=trainingsetindex,
        shuffle=shuffle,
        modelprefix=modelprefix,
    )


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
