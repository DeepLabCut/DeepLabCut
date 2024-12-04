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
from deeplabcut.gui.window import call_with_timeout

def test_call_with_timeout():
    def succeeding_method(parameter):
        return parameter

    parameter = (10, "Hello test")
    assert call_with_timeout(succeeding_method, 1, parameter) == parameter

    def failing_method():
        raise ValueError("Raise value error on purpose")

    with pytest.raises(ValueError):
        call_with_timeout(failing_method, timeout=1)

    def hanging_method():
        while True:
            time.sleep(1)

    with pytest.raises(TimeoutError):
        call_with_timeout(hanging_method, timeout=1)
