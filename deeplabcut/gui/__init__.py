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

import os
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

# Selection precedence:
#   1. Explicit, valid, user-set QT_API always wins;
#   2. Otherwise an already imported binding wins
#   3. Otherwise PySide6. qtpy's own default order (pyqt5, pyside2, pyqt6,
#      pyside6) puts PySide6 LAST, so an environment that also has PyQt6
#      (napari[all], for instance) would silently demote without this;
#   4. Otherwise leave QT_API unset and let qtpy autodetect.
_MODULE_TO_QT_API = (
    ("PyQt5", "pyqt5"),
    ("PySide2", "pyside2"),
    ("PyQt6", "pyqt6"),
    ("PySide6", "pyside6"),
)
_VALID_QT_APIS = tuple(api for _, api in _MODULE_TO_QT_API)
_GPL_QT_APIS = ("pyqt5", "pyqt6")

_NO_BINDING_MESSAGE = (
    "The DeepLabCut GUI requires a Qt binding, but none could be imported. "
    "PySide6 is the supported binding; install the GUI dependencies with "
    "`pip install 'deeplabcut[gui]'`. "
    "If PySide6 is already installed, it could not be loaded: on Linux this "
    "usually means the Qt system libraries are missing."
)

_user_qt_api = (os.environ.get("QT_API") or "").lower()
if _user_qt_api and _user_qt_api not in _VALID_QT_APIS:
    raise ValueError(
        f"QT_API is set to an unsupported value {_user_qt_api!r}. Valid values are: {', '.join(_VALID_QT_APIS)}."
    )

if not _user_qt_api:
    _already_loaded = next((api for module, api in _MODULE_TO_QT_API if module in sys.modules), None)
    if _already_loaded is not None:
        os.environ["QT_API"] = _already_loaded
    elif find_spec("PySide6") is not None:
        # find_spec does not import PySide6.
        # Let qtpy import so version checks and
        # its QT_API write-back still run
        os.environ["QT_API"] = "pyside6"

try:
    import qtpy  # noqa: F401  imported side effect: binding selection
except ImportError as err:
    raise ImportError(_NO_BINDING_MESSAGE) from err

# qtpy rewrites QT_API to the binding it actually loaded, but be explicit:
# matplotlib's backends/qt_compat.py also reads QT_API and must not end up on a
# different binding. It expects the lowercase form, which is `qtpy.API`.
os.environ["QT_API"] = qtpy.API

if qtpy.API in _GPL_QT_APIS:
    warnings.warn(
        f"DeepLabCut GUI is running on {qtpy.API_NAME}, which is licensed "
        "under the GPL. DeepLabCut is LGPL and is developed against PySide6 "
        "(LGPL). Set QT_API=pyside6, or install PySide6, to use the preferred "
        "binding.",
        stacklevel=2,
    )

BASE_DIR = Path(__file__).parent
