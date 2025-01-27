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
import pytest
import time
from deeplabcut.utils.multiprocessing import call_with_timeout


def _succeeding_method(parameter):
    return parameter


def _failing_method():
    raise ValueError("Raise value error on purpose")


def _hanging_method():
    while True:
        time.sleep(5)


def test_call_with_timeout():
    parameter = (10, "Hello test")
    assert call_with_timeout(_succeeding_method, 30, parameter) == parameter

    with pytest.raises(ValueError):
        call_with_timeout(_failing_method, timeout=30)

    with pytest.raises(TimeoutError):
        call_with_timeout(_hanging_method, timeout=1)
