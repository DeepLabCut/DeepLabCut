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
"""Small Matplotlib helpers shared across DeepLabCut."""

from __future__ import annotations

import logging

# Matplotlib builds this logger with ``logging.getLogger(__name__)`` in
# ``matplotlib/axes/_axes.py``, so the name is enough to reach it.
AXES_LOGGER_NAME = "matplotlib.axes._axes"


def silence_axes_logger(level: int | str = "ERROR") -> None:
    """Raise the log level of Matplotlib's ``Axes`` logger.

    Suppresses the chatty per-artist messages Matplotlib emits while drawing,
    most notably the invalid-color warnings raised during 3D plotting.

    Reaches the logger by name rather than importing ``_log`` from the private
    ``matplotlib.axes._axes`` module.  This is the same logger object, not a
    parallel one: Matplotlib creates it with ``logging.getLogger(__name__)``,
    and the logging manager returns that same instance here -- or, if this runs
    first, hands Matplotlib the instance created here.  Import order therefore
    does not matter.

    Args:
        level: Any level accepted by ``logging.Logger.setLevel``.
    """
    logging.getLogger(AXES_LOGGER_NAME).setLevel(level)
