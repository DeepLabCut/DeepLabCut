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

The fixture lives in ``tests/typing/top_level_api.py`` and is analyzed with
Pyright (or basedpyright). The check is skipped when neither is installed so it
never breaks a base CI job; run it where the checker is available to confirm
``deeplabcut/__init__.pyi`` resolves every lazy export to a real signature.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "typing" / "top_level_api.py"
_TYPE_CHECKERS = ("basedpyright", "pyright")


def _find_type_checker() -> str | None:
    for cmd in _TYPE_CHECKERS:
        if shutil.which(cmd):
            return cmd
    return None


@pytest.mark.skipif(_find_type_checker() is None, reason="No pyright/basedpyright available")
def test_top_level_api_resolves_statically() -> None:
    checker = _find_type_checker()
    proc = subprocess.run(
        [checker, str(_FIXTURE), "--outputjson"],
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)

    errors = [diagnostic for diagnostic in data.get("generalDiagnostics", []) if diagnostic.get("severity") == "error"]
    assert not errors, "\n".join(
        f"{e.get('file')}:{e.get('range', {}).get('start', {}).get('line', '?')}: {e.get('message')}" for e in errors
    )

    # ``reveal_type`` results must not degrade to ``Any`` or ``Unknown``, which
    # would mean the stub failed to expose the name statically.
    reveal_messages = [
        diagnostic.get("message", "")
        for diagnostic in data.get("generalDiagnostics", [])
        if 'is "' in diagnostic.get("message", "")
    ]
    assert reveal_messages, "expected reveal_type output from the type checker"
    for message in reveal_messages:
        assert "Unknown" not in message, message
        assert 'is "Any"' not in message, message
