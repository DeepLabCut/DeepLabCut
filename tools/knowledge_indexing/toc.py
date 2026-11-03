"""Read `_toc.yml`, the Jupyter Book table of contents.

The table of contents defines which pages are published — `docs/` also holds
pages it leaves out — and it is the only place the part / chapter / section
hierarchy exists, so it is what gives each page a parent and children.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TOC_FILE = "_toc.yml"

# Jupyter Book accepts `file:` entries with or without a suffix, and `_toc.yml`
# uses both spellings, so suffixes are normalised away on the way in.
SOURCE_SUFFIXES = (".md", ".ipynb", ".rst")


@dataclass(frozen=True)
class TocEntry:
    """One published page.

    `file` is repo-relative and has no suffix, exactly as `_toc.yml` spells it,
    e.g. `docs/installation`. Entries may point at notebooks as well as markdown.
    """

    file: str
    part: str = ""
    parent: str = ""
    children: tuple[str, ...] = ()


def read_toc(path: Path) -> list[TocEntry]:
    """Every page in the table of contents, in document order."""
    toc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries: list[TocEntry] = []

    root = toc.get("root")
    if root:
        entries.append(TocEntry(file=_normalize(root)))

    for part in toc.get("parts") or []:
        _collect(part.get("chapters") or [], part.get("caption") or "", "", entries)

    # A book may list chapters at the top level instead of grouping them in parts.
    _collect(toc.get("chapters") or [], "", "", entries)

    return entries


def _collect(items: list[dict[str, Any]], part: str, parent: str, entries: list[TocEntry]) -> None:
    """Append `items` and everything nested under them, depth first."""
    for item in items:
        if not item.get("file"):
            continue
        file = _normalize(item["file"])
        sections = item.get("sections") or []
        entries.append(
            TocEntry(
                file=file,
                part=part,
                parent=parent,
                children=tuple(_normalize(s["file"]) for s in sections if s.get("file")),
            )
        )
        _collect(sections, part, file, entries)


def _normalize(file: str) -> str:
    """Strip a source suffix so every entry is spelled the same way."""
    for suffix in SOURCE_SUFFIXES:
        if file.endswith(suffix):
            return file[: -len(suffix)]
    return file
