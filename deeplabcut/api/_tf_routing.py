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

import functools
import inspect
import warnings
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.core.engine import Engine

_TF_MODULE = "deeplabcut.tensorflow_compat"


@lru_cache
def _get_tensorflow_impl(name: str, module: str | None = None):
    mod = import_module(module or _TF_MODULE)
    return getattr(mod, name)


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


def _bind_unified_kwargs(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """Resolve all positional args to a unified kwargs dict using the function signature.

    For ``*args`` parameters, positional values are preserved in a ``_var_positional``
    key so they can be forwarded to the underlying implementation.
    """
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    unified: dict = {}
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            unified["_var_positional"] = bound.arguments.get(name, ())
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            unified.update(bound.arguments.get(name, {}))
        elif name in bound.arguments:
            unified[name] = bound.arguments[name]

    return unified


def with_tensorflow_fallback(
    _fn: Callable | None = None,
    *,
    tensorflow_name: str | None = None,
    renamed_params: dict[str, str] | None = None,
    dropped_params: list[str] | None = None,
    normalize_gputouse: bool = False,
    when: Callable[..., bool] | None = None,
    tensorflow_module: str | None = None,
) -> Callable:
    """Decorator for wrapping canonical PyTorch API functions, routing to a fallback TF function if required.

    By default, it resolves the engine from project configuration (via ``_resolve_engine``) and converts legacy
    TensorFlow kwargs to canonical PyTorch kwargs.  For functions that do not have a project config (e.g. modelzoo),
    a custom ``when`` callable can be supplied.

    Can be used with or without parentheses.

    Args:
        tensorflow_name (str | None): The name of the fallback TensorFlow function in ``tensorflow_module``. If not
            specified, uses the name of the canonical PyTorch function.
        renamed_params (dict[str, str] | None): Optional mapping from old TF parameter names to the new canonical
            PyTorch names. A warning will be emitted and the value is passed under the new canonical name. If both the
            old and new names are specified, raises a TypeError.
        dropped_params (list[str] | None): TF-only parameters that are silently removed before calling the canonical
            (PyTorch) function. A warning is emitted when they are dropped.
        normalize_gputouse (bool): resolve the old TF ``gputouse`` parameter to the new canonical PyTorch ``device``
            parameter. Raises a TypeError if both are specified.
        when (Callable | None): A callable ``(*args, **kwargs) -> bool`` that determines whether to route to the
            TensorFlow fallback.  When ``None`` (the default), the engine is resolved from shuffle metadata via
            ``_resolve_engine``.  Supply a custom callable for engine-less routing (e.g. modelzoo functions).
        tensorflow_module (str | None): Override the module from which to import the TF fallback function. Defaults to
            ``"deeplabcut.tensorflow_compat"``.

    Note:
        When ``when`` is ``None``, the engine is resolved from the shuffle metadata if not specified explicitly. If
        neither ``shuffles``, ``shuffle`` nor ``engine`` is passed, it assumes shuffle=1.
    """

    def decorator(fn):
        tf_name = tensorflow_name or fn.__name__
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            unified = _bind_unified_kwargs(sig, args, kwargs)

            if when is not None:
                # Custom condition routing (e.g. modelzoo functions)
                route_to_tf = when(*args, **kwargs)
            else:
                # Default: engine-based routing (from shuffle / config)
                route_to_tf = _resolve_engine(unified) == Engine.TF

            # Strip engine before forwarding to either implementation
            unified.pop("engine", None)

            if route_to_tf:
                warn_deprecated_tensorflow()
                return _get_tensorflow_impl(tf_name, module=tensorflow_module)(
                    *args,
                    **{k: v for k, v in kwargs.items() if k != "engine"},
                )

            resolved = _resolve_legacy_kwargs(
                unified,
                renamed_params=renamed_params or {},
                dropped_params=dropped_params or [],
                normalize_gputouse=normalize_gputouse,
            )

            var_positional = resolved.pop("_var_positional", ())
            if var_positional:
                return fn(*var_positional, **resolved)
            return fn(**resolved)

        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator


def _shuffles_from_kwargs(kwargs: dict) -> list | tuple:
    """Return shuffle indices from kwargs, accepting legacy ``Shuffles``."""
    if "shuffles" in kwargs and "Shuffles" in kwargs:
        raise TypeError("Cannot specify both 'Shuffles' (deprecated) and 'shuffles'. Use 'shuffles' only.")
    if "shuffles" in kwargs:
        return kwargs["shuffles"]
    if "Shuffles" in kwargs:
        return kwargs["Shuffles"]
    return [kwargs.get("shuffle", 1)]


def _resolve_engine(unified_kwargs: dict) -> Engine:
    """Resolve engine from explicit engine parameter or shuffle metadata.

    Args:
        unified_kwargs: Keyword arguments resolved from positional and keyword args
                        via ``_bind_unified_kwargs``.
    """
    engine = unified_kwargs.get("engine")
    if engine is not None:
        return engine

    from deeplabcut.core.config.utils import read_config

    shuffles = _shuffles_from_kwargs(unified_kwargs)
    config = unified_kwargs["config"]
    cfg = read_config(config)
    from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine

    engines = {
        get_shuffle_engine(
            cfg,
            trainingsetindex=unified_kwargs.get("trainingsetindex", 0),
            shuffle=s,
            modelprefix=unified_kwargs.get("modelprefix", ""),
        )
        for s in shuffles
    }
    if len(engines) > 1:
        raise ValueError(f"All shuffles must have the same engine (found different engines for shuffles: {shuffles}).")
    return engines.pop()


def _normalize_gputouse(gputouse: str | int) -> str:
    if isinstance(gputouse, int):
        return f"cuda:{gputouse}"
    if gputouse.startswith("cuda:"):
        return gputouse
    if gputouse.startswith("gpu:"):
        return gputouse.replace("gpu:", "cuda:")
    return gputouse


def _resolve_legacy_kwargs(
    kwargs: dict,
    renamed_params: dict[str, str],
    dropped_params: list[str],
    normalize_gputouse: bool = False,
) -> dict:
    """Resolve legacy TensorFlow kwargs to canonical (PyTorch) kwargs."""

    if normalize_gputouse and (gpu := kwargs.get("gputouse")):
        # Normalize parameter "gputouse" to torch device string and rename
        kwargs["gputouse"] = _normalize_gputouse(gpu)
        renamed_params["gputouse"] = "device"

    # Rename deprecated parameters
    for old, new in renamed_params.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError(f"Cannot specify both '{old}' (deprecated) and '{new}'. Use '{new}' only.")
            kwargs[new] = kwargs.pop(old)
            warnings.warn(
                f"'{old}' is deprecated; use {new}='{kwargs[new]}' instead.",
                DLCDeprecationWarning,
                stacklevel=3,
            )

    # Drop unused parameters
    for key in dropped_params:
        if key in kwargs:
            kwargs.pop(key)
            warnings.warn(
                f"'{key}' is a TensorFlow-only parameter and has no effect for PyTorch projects.",
                DLCDeprecationWarning,
                stacklevel=3,
            )
    return kwargs
