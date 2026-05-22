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

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    Accepted truthy values:
    1, true, yes, on

    Accepted falsy values:
    0, false, no, off
    """
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    """Parse an optional float environment variable.

    Empty strings / unset values return ``default``.
    Invalid values also fall back to ``default``.
    """
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _which(command: str) -> str:
    try:
        resolved = shutil.which(command)
        return str(Path(resolved).resolve()) if resolved else "not-found"
    except Exception:
        return "not-found"


def _command_version(command: str, version_args: Sequence[str] = ("-version",)) -> str:
    try:
        completed = subprocess.run(
            [command, *version_args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return "unavailable"

    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return "unavailable"

    first_line = text.splitlines()[0].strip()
    return first_line or "unavailable"
