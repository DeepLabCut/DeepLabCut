#!/usr/bin/env python3
"""Generate a readable Markdown report from Ruff JSON output.

Usage:
    python ruff_report.py . --output ruff-report.md
    python ruff_report.py src tests --output lint/ruff-report.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

RULE_NOTES = {
    "F401": "Unused import. Usually safe to delete; verify imports with side effects.",
    "E501": "Line too long. Prefer wrapping expressions, splitting long strings/comments, or extracting variables.",
    "E402": "Module import not at top of file. Move imports above executable code if possible.",
    "F403": "`from x import *` makes names unclear. Replace with explicit imports.",
    "F405": "Likely consequence of `import *`. Import the name explicitly.",
    "F821": "Undefined name. Usually a real bug or missing import.",
    "E722": "Bare `except:`. Catch `Exception` or a narrower exception type.",
    "B904": "Inside `except`, use `raise ... from e` to preserve exception chaining.",
    "B007": "Unused loop variable. Rename to `_` or use it.",
    "UP031": "Old `%` formatting. Convert to f-strings or `.format()` where appropriate.",
    "E721": "Avoid direct `type(x) == Y`; prefer `isinstance(x, Y)`.",
    "B008": "Function call in default arg. Use `None` + initialize inside the function.",
    "B023": "Function closes over loop variable. Bind it via default arg or helper.",
    "B024": "ABC without abstract method. Add `@abstractmethod` or remove ABC intent.",
    "F811": "Redefined while unused. Remove duplicate or rename.",
    "B012": "Jump statement in `finally` can swallow exceptions. Restructure flow.",
    "B016": "Raise an exception instance/class, not a literal.",
    "B017": "Use a more specific exception with `assertRaises`.",
    "B020": "Loop variable overrides iterator. Rename loop variables.",
    "B027": "Empty method in ABC without abstract decorator. Add `@abstractmethod` or implement it.",
}


def run_ruff(paths: Iterable[str]) -> list[dict]:
    cmd = [sys.executable, "-m", "ruff", "check", *paths, "--output-format=json", "--exit-zero"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"Failed to run Ruff: {' '.join(cmd)}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise SystemExit("Unexpected Ruff JSON output")
    return data


def relpath(path: str) -> str:
    try:
        return os.path.relpath(path)
    except Exception:
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["."], help="Files/directories to scan")
    parser.add_argument("--output", default="tmp/ruff-report.md", help="Markdown output path")
    args = parser.parse_args()

    issues = run_ruff(args.paths)

    by_rule: dict[str, list[dict]] = collections.defaultdict(list)
    for item in issues:
        by_rule[item.get("code", "UNKNOWN")].append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Ruff manual-fix report\n")
    lines.append(f"Generated from: `{', '.join(args.paths)}`\n")
    lines.append(f"Total remaining issues: **{len(issues)}**\n")

    lines.append("## Summary\n")
    lines.append("| Rule | Count | Note |")
    lines.append("|---|---:|---|")
    for rule, items in sorted(by_rule.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        note = RULE_NOTES.get(rule, "")
        lines.append(f"| `{rule}` | {len(items)} | {note} |")
    lines.append("")

    lines.append("## Suggested triage order\n")
    preferred = ["F403", "F405", "F821", "E722", "B904", "E402", "F401", "E501"]
    present = [r for r in preferred if r in by_rule]
    if present:
        for idx, rule in enumerate(present, 1):
            lines.append(f"{idx}. `{rule}` — {RULE_NOTES.get(rule, '')}")
        lines.append("")

    lines.append("## Table of contents by rule\n")
    for rule, items in sorted(by_rule.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        anchor = rule.lower()
        lines.append(f"- [{rule} ({len(items)})](#{anchor})")
    lines.append("")

    for rule, items in sorted(by_rule.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"## {rule}\n")
        lines.append(f"Count: **{len(items)}**  ")
        if rule in RULE_NOTES:
            lines.append(f"Hint: {RULE_NOTES[rule]}  ")
        lines.append("")

        file_groups: dict[str, list[dict]] = collections.defaultdict(list)
        for item in items:
            file_groups[relpath(item["filename"])].append(item)

        lines.append("### Files affected\n")
        lines.append("| File | Count |")
        lines.append("|---|---:|")
        for filename, entries in sorted(file_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"| `{filename}` | {len(entries)} |")
        lines.append("")

        lines.append("### Details\n")
        for filename, entries in sorted(file_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"#### `{filename}` ({len(entries)})\n")
            lines.append("| Line | Col | Message |")
            lines.append("|---:|---:|---|")
            for e in sorted(
                entries, key=lambda x: (x.get("location", {}).get("row", 0), x.get("location", {}).get("column", 0))
            ):
                loc = e.get("location", {})
                line = loc.get("row", "")
                col = loc.get("column", "")
                msg = (e.get("message", "") or "").replace("|", "\\|")
                lines.append(f"| {line} | {col} | {msg} |")
            lines.append("")
            lines.append("Quick open commands:")
            lines.append("")
            lines.append("```powershell")
            lines.append(f'code -g "{filename}:{entries[0].get("location", {}).get("row", 1)}"')
            lines.append("```")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
