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
"""Fixtures for public API / TensorFlow routing tests."""

from __future__ import annotations

import pytest

from deeplabcut.api import _tf_routing as tf_routing


@pytest.fixture(autouse=True)
def reset_tf_deprecation_warning():
    """Reset the once-per-process TF banner latch between tests.

    ``warn_deprecated_tensorflow`` intentionally emits only once per process so
    dependency warning-filter churn cannot re-spam the banner. Tests that assert
    emission need a clean latch each time.
    """
    tf_routing._TF_DEPRECATION_WARNED = False
    yield
    tf_routing._TF_DEPRECATION_WARNED = False
