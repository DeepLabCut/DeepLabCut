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

# A missing optional dependency surfaces as a plain ``ModuleNotFoundError``
# naming the module, which is accurate and points at the failing import. The
# ``dlc`` entry point (see ``__main__.py``) is where a user without the GUI
# extra actually lands, and it already tells them to install ``deeplabcut[gui]``.
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
__all__ = [*__all__, "DEBUG"]
