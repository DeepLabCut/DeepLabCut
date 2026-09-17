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
"""Tests for ``deeplabcut.utils.matplotlib_utils``."""

from __future__ import annotations

import logging

import pytest

from deeplabcut.utils import matplotlib_utils as mu


@pytest.fixture
def axes_logger():
    logger = logging.getLogger(mu.AXES_LOGGER_NAME)
    original = logger.level
    yield logger
    logger.setLevel(original)


def test_axes_logger_name_matches_matplotlibs_own_logger():
    """``AXES_LOGGER_NAME`` must resolve to the logger Matplotlib actually uses.

    ``silence_axes_logger`` reaches that logger by name instead of importing
    ``_log``, so a module rename upstream would silently stop suppressing
    messages rather than raising.  This is the assertion that turns such a
    rename back into a loud failure, and the one place where importing the
    private module is the point.
    """
    from matplotlib.axes._axes import _log

    assert logging.getLogger(mu.AXES_LOGGER_NAME) is _log


def test_silence_axes_logger_sets_the_level(axes_logger):
    axes_logger.setLevel(logging.DEBUG)

    mu.silence_axes_logger()

    assert axes_logger.level == logging.ERROR
