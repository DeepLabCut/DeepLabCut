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
from importlib import import_module
from typing import Any

from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.core.engine import Engine

_TF_MODULE = "deeplabcut.tensorflow_compat"


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


def _apply_parameter_renames(
    parameters: dict[str, Any],
    renames: dict[str, str],
    *,
    warn: bool = False,
    stacklevel: int = 4,
) -> None:
    """Rename deprecated keys in ``kwargs`` to their canonical names, in place.
    Raises a ``TypeError`` when both the deprecated and canonical names are given.
    Args:
        parameters: Keyword-argument dict to mutate.
        renames: Mapping of deprecated names to canonical names.
        warn: If ``True``, emit a ``DLCDeprecationWarning`` for each rename.
    """
    for old, new in renames.items():
        if old not in parameters:
            continue
        if new in parameters:
            raise TypeError(f"Cannot specify both '{old}' (deprecated) and '{new}'. Use '{new}' only.")
        parameters[new] = parameters.pop(old)
        if warn:
            warnings.warn(
                f"'{old}' is deprecated; use {new}={parameters[new]!r} instead.",
                DLCDeprecationWarning,
                stacklevel=stacklevel,
            )


def _positionals_as_kwargs(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """Routing view: kwargs plus positionals mapped to parameter names.

    Excess positionals captured by a variadic ``*args`` parameter are included
    as a tuple under the variadic parameter's name (e.g. `args`).

    Does not validate unknown kwargs — legacy aliases are forwarded downstream
    and handled by ``@renamed_parameter`` / backend-specific logic.
    """
    unified = dict(kwargs)
    positional = []
    var_positional = None
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional.append(param)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = param

    if len(args) > len(positional) and var_positional is None:
        raise TypeError(f"too many positional arguments (expected at most {len(positional)}, got {len(args)})")
    for param, value in zip(positional, args, strict=False):
        if param.name in unified:
            raise TypeError(f"got multiple values for argument '{param.name}'")
        unified[param.name] = value
    if var_positional is not None:
        unified[var_positional.name] = args[len(positional) :]
    return unified


def with_tensorflow_fallback(
    _fn: Callable | None = None,
    *,
    tensorflow_name: str | None = None,
    renamed_params: dict[str, str] | None = None,
    dropped_params: list[str] | None = None,
    normalize_gputouse: bool = False,
    when: Callable[[dict[str, Any]], bool] | None = None,
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
            Note: applied **only on the PyTorch path**. The TF compat functions in ``tensorflow_compat/`` maintain their
            own legacy parameter names and are not affected by this mapping.)
        dropped_params (list[str] | None): TF-only parameters that are silently removed before calling the canonical
            (PyTorch) function. A warning is emitted when they are dropped.
            Note: applied **only on the PyTorch path**. The TF compat functions accept these parameters natively.
        normalize_gputouse (bool): Resolve the old TF ``gputouse`` parameter to the new canonical PyTorch ``device``
            parameter. Raises a TypeError if both are specified. Equivalent to ``renamed_params={"gputouse": "device"}``
            with the additional normalization of legacy formats (``int``, ``"gpu:0"``) to ``"cuda:X"``.
            Note: applied **only on the PyTorch path**. When ``True``, setting ``gputouse`` in ``dropped_params`` is
            redundant (``normalize_gputouse`` always renames ``gputouse`` to ``device`` first).
        when (Callable[[dict[str, Any]], bool] | None): A callable ``(kwargs: dict[str, Any]) -> bool`` that
            determines whether to route to the TensorFlow fallback. It receives the bound keyword arguments of the
            wrapped function (positionals mapped to their parameter names). When ``None`` (the default), the engine is
            resolved from shuffle metadata via ``_resolve_engine``. Supply a custom callable for engine-less routing
            (e.g. modelzoo functions).
        tensorflow_module (str | None): Override the module from which to import the TF fallback function. Defaults to
            ``"deeplabcut.tensorflow_compat"``.

    Note:
        When ``when`` is ``None``, the engine is resolved from the shuffle metadata if not specified explicitly. If
        neither ``shuffles``, ``shuffle`` nor ``engine`` is passed, it assumes shuffle=1.

        The original ``*args`` / ``**kwargs`` (minus ``engine``) are forwarded downstream. Parameter renames such as
        ``displayiters`` → ``display_iters`` stay on ``@renamed_parameter`` / the TF backend, not in this router.

        Legacy cleanup (``renamed_params``, ``dropped_params``, ``normalize_gputouse``) is applied only when the
        PyTorch path is taken. The TF path forwards arguments as-is so the TF compat functions receive their
        native parameter names.
    """

    def decorator(fn):
        tf_name = tensorflow_name or fn.__name__
        sig = inspect.signature(fn)

        # ``engine`` is a routing-only parameter that is consumed by this router
        # it MUST be keyword-only to prevent leaking it to the delegate functions
        engine_param = sig.parameters.get("engine")
        if engine_param is not None and engine_param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{fn.__qualname__}: 'engine' must be a keyword-only parameter")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Acquire all arguments as keyword arguments with canonical names for the TF routing decision
            unified = _positionals_as_kwargs(sig, args, kwargs)
            _apply_parameter_renames(unified, renames=renamed_params or {})

            if when is not None:
                # Custom condition routing (e.g. modelzoo functions)
                route_to_tf = when(unified)
            else:
                # Default: engine-based routing (from shuffle / config)
                route_to_tf = _resolve_engine(unified) == Engine.TF

            kwargs.pop("engine", None)

            if route_to_tf:
                warn_deprecated_tensorflow()
                return _get_tensorflow_impl(tf_name, module=tensorflow_module)(
                    *args,
                    **kwargs,
                )

            # PT-only: router-owned legacy cleanup (allow_growth, keepdeconvweights, …)
            kwargs = _resolve_legacy_kwargs(
                kwargs,
                renamed_params=renamed_params or {},
                dropped_params=dropped_params or [],
                normalize_gputouse=normalize_gputouse,
            )
            # Inner @renamed_parameter still sees aliases like displayiters / videotype
            return fn(*args, **kwargs)

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
        unified_kwargs: Keyword arguments with positionals mapped to names via
                        ``_positionals_as_kwargs``.
    """
    engine = unified_kwargs.get("engine")
    if engine is not None:
        return Engine(engine)

    from deeplabcut.core.config.utils import read_config

    shuffles = _shuffles_from_kwargs(unified_kwargs)
    if not shuffles:
        raise ValueError("Shuffles must contain at least one index")
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


def _normalize_gputouse(gputouse: str | int | None) -> str | None:
    if isinstance(gputouse, int):
        return f"cuda:{gputouse}"
    if gputouse is None or gputouse.startswith("cuda:"):
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

    effective_renames = dict(renamed_params)

    if normalize_gputouse and "gputouse" in kwargs:
        # Normalize parameter "gputouse" to torch device string and rename
        kwargs["gputouse"] = _normalize_gputouse(kwargs["gputouse"])
        effective_renames["gputouse"] = "device"

    # Rename deprecated parameters
    _apply_parameter_renames(kwargs, renames=effective_renames, warn=True)

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
