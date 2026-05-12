from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


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
