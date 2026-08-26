"""Serialise the knowledge index to disk.

The shapes written here are defined in `schemas.py`; this module only decides
where each one goes and how it is dumped.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .schemas import MANIFEST, SYMBOL_TABLE, ApiNode, Manifest, Node, SymbolTable


def write_index(
    output_dir: Path,
    groups: dict[str, Sequence[Node]],
    version: str,
    revision: str = "",
    base_urls: dict[str, str] | None = None,
) -> None:
    """Write `groups` (directory name -> nodes) under `output_dir`, and the manifest."""
    for directory, nodes in groups.items():
        _check_unique_ids(directory, nodes)
        node_dir = output_dir / directory
        node_dir.mkdir(parents=True, exist_ok=True)
        for node in nodes:
            _write_yaml(node_dir / node.filename, node.to_dict())

    manifest = Manifest(
        version=version,
        generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        nodes=groups,
        revision=revision,
        base_urls=base_urls or {},
    )
    _write_yaml(output_dir / MANIFEST, manifest.to_dict())


def _check_unique_ids(directory: str, nodes: Sequence[Node]) -> None:
    """Raise if `nodes` has a duplicate id, which would silently overwrite a file."""
    seen: set[str] = set()
    for node in nodes:
        if node.id in seen:
            raise ValueError(f"Duplicate node id {node.id!r} in {directory}/")
        seen.add(node.id)


def write_symbol_table(output_dir: Path, apis: Sequence[ApiNode]) -> int:
    """Write the symbol table for `apis`, returning the number of symbols in it."""
    table = SymbolTable.from_apis(apis)
    _write_yaml(output_dir / SYMBOL_TABLE, table.to_dict())
    return len(table.symbols)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Serialise `data` to `path`, preserving the field order of the schema."""
    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
