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
"""Runs a focused type-checker smoke test over the top-level public API.

The check itself lives in ``tools/check_type_stub.py`` so CI can run it without
installing DeepLabCut (see the ``typecheck`` job in ``.github/workflows/format.yml``).
This test is the local entry point: run it with pytest to confirm
``deeplabcut/__init__.pyi`` resolves every lazy export to a real signature. It is
skipped when neither Pyright nor basedpyright is installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "check_type_stub.py"
_NO_CHECKER = 2


def test_top_level_api_resolves_statically() -> None:
    proc = subprocess.run([sys.executable, str(_TOOL)], capture_output=True, text=True)

    if proc.returncode == _NO_CHECKER:
        pytest.skip(proc.stdout.strip() or "no pyright/basedpyright available")

    assert proc.returncode == 0, proc.stdout + proc.stderr
