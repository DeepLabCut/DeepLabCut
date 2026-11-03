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
from __future__ import annotations

import logging

AXES_LOGGER_NAME = "matplotlib.axes._axes"


def silence_axes_logger(level: int | str = "ERROR") -> None:
    """Raise the log level of Matplotlib's ``Axes`` logger.

    Suppresses the chatty per-artist messages Matplotlib emits while drawing,
    most notably the invalid-color warnings raised during 3D plotting.

    Args:
        level: Any level accepted by ``logging.Logger.setLevel``.
    """
    logging.getLogger(AXES_LOGGER_NAME).setLevel(level)
