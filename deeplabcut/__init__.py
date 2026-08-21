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

import logging
import os
import warnings

import lazy_loader as lazy

from deeplabcut.core.deprecation import DLCDeprecationWarning

from .version import VERSION, __version__

logger = logging.getLogger(__name__)

# DEBUG="", "0", "false", "no" -> False
DEBUG = os.environ.get("DEBUG", "").strip().lower() not in {"", "0", "false", "no"}

if DEBUG:
    logger.debug("Loading DLC %s", VERSION)

# DeepLabCut deprecation warnings are shown only once per message instance.
warnings.filterwarnings("once", category=DLCDeprecationWarning)

# -----------------------------------------------------------------------------
# Stub-driven lazy loading
# -----------------------------------------------------------------------------
# ``deeplabcut/__init__.pyi`` is the single declarative source of truth for the
# top-level public API. ``lazy_loader.attach_stub`` reads it at runtime to
# install ``__getattr__``, ``__dir__``, and ``__all__``, so each implementation
# module is imported only when its top-level attribute is first accessed.
# -----------------------------------------------------------------------------

_lazy_getattr, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# -----------------------------------------------------------------------------
# Optional-dependency diagnostics
# -----------------------------------------------------------------------------
# A plain ``attach_stub`` raises ``ModuleNotFoundError`` when a GUI or PyTorch
# tracking module is unavailable. Translate only those into actionable
# ``ImportError`` messages and leave unrelated import failures untouched.
# -----------------------------------------------------------------------------

_GUI_EXPORTS = frozenset(
    {
        "launch_dlc",
        "label_frames",
        "refine_labels",
        "refine_tracklets",
        "SkeletonBuilder",
    }
)

_TORCH_EXPORTS = frozenset({"transformer_reID"})

_GUI_DEPENDENCY_MODULES = frozenset({"PySide6", "napari", "qdarkstyle"})
_TORCH_DEPENDENCY_MODULES = frozenset({"torch", "torchvision"})


def _is_missing_gui_dependency(exc: ModuleNotFoundError) -> bool:
    """Return True if ``exc`` is caused by a missing GUI dependency."""
    name = getattr(exc, "name", None)
    return isinstance(name, str) and name.split(".")[0] in _GUI_DEPENDENCY_MODULES


def _is_missing_torch_dependency(exc: ModuleNotFoundError) -> bool:
    """Return True if ``exc`` is caused by a missing PyTorch dependency."""
    name = getattr(exc, "name", None)
    return isinstance(name, str) and name.split(".")[0] in _TORCH_DEPENDENCY_MODULES


def __getattr__(name: str):
    try:
        return _lazy_getattr(name)
    except ModuleNotFoundError as exc:
        if name in _GUI_EXPORTS and _is_missing_gui_dependency(exc):
            raise ImportError(
                f"{name!r} requires the DeepLabCut GUI dependencies. Install the supported GUI extra."
            ) from exc

        if name in _TORCH_EXPORTS and _is_missing_torch_dependency(exc):
            raise ImportError(f"{name!r} requires the PyTorch tracking dependencies.") from exc

        raise
