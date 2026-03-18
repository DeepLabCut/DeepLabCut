#!/usr/bin/env python3
"""Reduce Ruff E501 violations using autopep8, then normalize with Ruff.

Usage:
    python fix_e501_with_autopep8.py . --line-length 88
    python fix_e501_with_autopep8.py src tests --line-length 100 --check

NOTE: if this creates broken escaped f-strings :
f"some string with a {
    var
}"
Use the ^[ \t]*\}"[ \t]*$ regex to find and fix them.

Requirements:
    - ruff
    - autopep8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def ruff_json(paths: list[str]) -> list[dict]:
    proc = run(["ruff", "check", *paths, "--output-format=json", "--exit-zero"], check=False)
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Could not parse Ruff JSON: {e}") from e
    if not isinstance(data, list):
        raise SystemExit("Unexpected Ruff JSON output")
    return data


def unique_e501_files(paths: list[str]) -> list[str]:
    data = ruff_json(paths)
    files = sorted({item["filename"] for item in data if item.get("code") == "E501"})
    return files


def chunked(items: list[str], size: int = 50):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["."], help="Files/directories to scan")
    parser.add_argument("--line-length", type=int, default=88)
    parser.add_argument("--check", action="store_true", help="Dry run; only show affected files")
    args = parser.parse_args()

    files = unique_e501_files(args.paths)
    if not files:
        print("No E501 files found. Nothing to do.")
        return 0

    print(f"Found {len(files)} file(s) with E501.")
    for f in files:
        print(" -", f)

    if args.check:
        return 0

    for batch in chunked(files, 50):
        run(
            [
                "autopep8",
                "--in-place",
                "--aggressive",
                f"--max-line-length={args.line_length}",
                "--select=E501,W291,W292,W391",
                *batch,
            ]
        )

        run(["ruff", "check", *batch, "--fix", "--unsafe-fixes"], check=False)
        run(["ruff", "format", *batch], check=False)

    after = len(unique_e501_files(args.paths))
    print(f"Remaining files with E501: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
