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
"""Check that ``deeplabcut/__init__.pyi`` exposes the public API to type checkers.

Runs Pyright over ``tools/typing/top_level_api.py`` and verifies that every
``reveal_type`` resolves to a real declaration rather than ``Unknown``/``Any``.
That file is an input to this tool, not a pytest module, so it lives here beside
the tool rather than under ``tests/``.

Runs without DeepLabCut or its dependencies installed, reading the source tree
directly, which is what keeps it cheap enough for CI. That is also the right
scope: the stub's job is to expose the *names* statically, while resolving the
third-party types inside a signature depends on those packages being present.

Exit codes: 0 sound, 1 failures, 2 no type checker found (unless
``--require-checker``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tools" / "typing" / "top_level_api.py"
TYPE_CHECKERS = ("basedpyright", "pyright")

NO_CHECKER = 2


def find_type_checker() -> str | None:
    """Return the first available type checker, or None."""
    for command in TYPE_CHECKERS:
        if shutil.which(command):
            return command
    return None


def check(checker: str, fixture: Path = FIXTURE) -> list[str]:
    """Return human-readable failures; an empty list means the stub is sound."""
    proc = subprocess.run(
        [checker, str(fixture), "--outputjson"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = "\n".join((proc.stderr or proc.stdout).splitlines()[:5])
        return [
            f"{checker} produced no JSON report (exit {proc.returncode}). "
            f"It runs on Node, so check that a current `node` is on PATH.\n{detail}"
        ]

    # Only diagnostics for the fixture itself; a missing third-party package in
    # some transitively-read module is not this check's business.
    diagnostics = [d for d in data.get("generalDiagnostics", []) if Path(d.get("file", "")).name == fixture.name]

    failures = [
        f"{Path(d.get('file', '?')).name}:{d.get('range', {}).get('start', {}).get('line', '?')}: {d.get('message')}"
        for d in diagnostics
        if d.get("severity") == "error"
    ]

    reveals = [d.get("message", "") for d in diagnostics if 'is "' in d.get("message", "")]
    if not reveals:
        failures.append(f"no reveal_type output from {checker}; the fixture may not have been analyzed")

    # A name the stub failed to expose degrades to exactly Unknown or Any.
    # Nested Unknowns are expected here, since dependencies are not installed.
    failures += [m for m in reveals if 'is "Unknown"' in m or 'is "Any"' in m]

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-checker",
        action="store_true",
        help=f"fail instead of exiting {NO_CHECKER} when no type checker is installed",
    )
    args = parser.parse_args()

    checker = find_type_checker()
    if checker is None:
        message = f"no type checker found (tried: {', '.join(TYPE_CHECKERS)})"
        if args.require_checker:
            print(f"error: {message}", file=sys.stderr)
            return 1
        print(f"skipped: {message}")
        return NO_CHECKER

    failures = check(checker)
    if failures:
        print(f"{checker} found {len(failures)} problem(s) with the public API stub:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"{checker}: deeplabcut/__init__.pyi resolves the public API statically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
