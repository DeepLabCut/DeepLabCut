#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""The process-wide ``DEBUG`` flag, re-exported as ``deeplabcut.DEBUG``.

Read once at import time, matching the historical behaviour: callers such as
``create_project`` treat it as a constant for the life of the process.
"""

from __future__ import annotations

import os

_DISABLED = {"", "0", "false", "no"}

#: True when the DEBUG environment variable is set to anything but "", 0, false or no.
DEBUG: bool = os.environ.get("DEBUG", "").strip().lower() not in _DISABLED
