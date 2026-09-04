"""Serialise the knowledge index to disk.

The record and manifest shapes written here are defined in `schemas.py`; this
module flattens the build-time trees from `api_index.py` and `docs_index.py`
into those records and writes them as JSON / JSONL.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schemas import (
    API_FILE,
    DOCS_FILE,
    TOP_MANIFEST,
    VERSION_MANIFEST,
    ApiProvenance,
    ApiRecord,
    DocPageRecord,
    DocSectionRecord,
    DocsProvenance,
    TopManifest,
    VersionManifest,
)

if TYPE_CHECKING:
    from .api_index import ApiNode
    from .docs_index import DocsPageNode


def write_version(
    knowledge_dir: Path,
    version_label: str,
    apis: Sequence[ApiNode] | None,
    docs_pages: Sequence[DocsPageNode] | None,
    package_version: str = "",
    revision: str = "",
) -> tuple[int, int]:
    """Write `api.jsonl` and/or `docs.jsonl` for one version, plus its manifest.

    `apis` (or `docs_pages`) may be None to leave that half untouched: its
    file and manifest provenance are kept exactly as already on disk under
    `knowledge_dir/version_label`. Returns the number of api and docs records
    written this run (0 for an untouched half).
    """
    version_dir = knowledge_dir / version_label
    version_dir.mkdir(parents=True, exist_ok=True)

    existing = _read_json(version_dir / VERSION_MANIFEST)
    api_provenance = ApiProvenance.from_dict(existing["api"]) if existing and existing.get("api") else None
    docs_provenance = DocsProvenance.from_dict(existing["docs"]) if existing and existing.get("docs") else None

    api_count = 0
    if apis is not None:
        api_records = _api_records(apis)
        _check_unique_ids(API_FILE, (record.id for record in api_records))
        _write_jsonl(version_dir / API_FILE, (record.to_dict() for record in api_records))
        api_count = len(api_records)
        api_provenance = ApiProvenance(
            package_version=package_version,
            revision=revision,
            generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        )

    docs_count = 0
    if docs_pages is not None:
        page_records, section_records = _doc_records(docs_pages)
        docs_records: list[DocPageRecord | DocSectionRecord] = [*page_records, *section_records]
        _check_unique_ids(DOCS_FILE, (record.id for record in docs_records))
        _write_jsonl(version_dir / DOCS_FILE, (record.to_dict() for record in docs_records))
        docs_count = len(docs_records)
        docs_provenance = DocsProvenance(
            package_version=package_version,
            revision=revision,
            generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        )

    if api_provenance is None:
        raise ValueError(f"No api provenance for {version_label!r}: apis was skipped and no manifest.json exists yet")

    manifest = VersionManifest(api_version_label=version_label, api=api_provenance, docs=docs_provenance)
    _write_json(version_dir / VERSION_MANIFEST, manifest.to_dict())

    return api_count, docs_count


def write_top_manifest(knowledge_dir: Path, docs_version_label: str) -> None:
    """Rebuild `knowledge/manifest.json` from the version directories under `knowledge_dir`."""
    versions = sorted(child.name for child in knowledge_dir.iterdir() if child.is_dir() and _has_api(child))
    has_docs = (knowledge_dir / docs_version_label / DOCS_FILE).is_file()
    manifest = TopManifest(
        docs_path=f"{docs_version_label}/{DOCS_FILE}" if has_docs else "",
        api_latest=docs_version_label,
        api_versions=tuple(versions),
    )
    _write_json(knowledge_dir / TOP_MANIFEST, manifest.to_dict())


def delete_version(knowledge_dir: Path, version_label: str) -> None:
    """Remove a released version's API index (only releases; "main" cannot be deleted)."""
    shutil.rmtree(knowledge_dir / version_label, ignore_errors=True)


def _has_api(version_dir: Path) -> bool:
    """True if `version_dir` publishes an api.jsonl the manifest vouches for."""
    if not (version_dir / API_FILE).is_file():
        return False
    manifest = _read_json(version_dir / VERSION_MANIFEST)
    return bool(manifest and manifest.get("api"))


def _doc_records(pages: Sequence[DocsPageNode]) -> tuple[list[DocPageRecord], list[DocSectionRecord]]:
    """Flatten each page and its sections into their published rows."""
    page_records = []
    section_records = []
    for page in pages:
        page_records.append(
            DocPageRecord(
                id=page.id,
                title=page.title,
                url=page.docs_url,
                source_file=page.source_file,
                section=page.part,
                summary=page.summary,
                status=page.status,
                last_verified=page.last_verified,
                parent=page.parent,
                children=page.children,
                related_pages=page.related_pages,
                labels=page.labels,
            )
        )
        for section in page.sections:
            section_records.append(
                DocSectionRecord(
                    id=section.id,
                    page=page.id,
                    title=section.title,
                    url=section.docs_url,
                    anchor=section.anchor,
                    level=section.level,
                    section=page.part,
                    summary=section.excerpt,
                )
            )
    return page_records, section_records


def _api_records(apis: Sequence[ApiNode]) -> list[ApiRecord]:
    """Flatten each module and its symbols into their published rows.

    A module gets its own row only if it has a docstring -- a module with
    neither a docstring nor symbols was already dropped by `build_api_nodes`,
    but one with only undocumented symbols would otherwise get an empty row.
    """
    records = []
    for node in apis:
        if node.summary:
            records.append(
                ApiRecord(
                    id=node.id,
                    kind="module",
                    name=node.module,
                    module=node.module,
                    url=node.docs_url,
                    summary=node.summary,
                    source=node.source,
                )
            )
        for symbol in node.symbols:
            records.append(
                ApiRecord(
                    id=f"{node.id}.{symbol.name}",
                    kind=symbol.kind,
                    name=symbol.name,
                    module=node.module,
                    url=symbol.docs_url,
                    signature=symbol.signature,
                    summary=symbol.summary,
                    source=symbol.source,
                )
            )
    return records


def _check_unique_ids(filename: str, ids: Iterable[str]) -> None:
    """Raise if `ids` has a duplicate, which would otherwise silently collide."""
    seen: set[str] = set()
    for record_id in ids:
        if record_id in seen:
            raise ValueError(f"Duplicate record id {record_id!r} in {filename}")
        seen.add(record_id)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a single, human-readable JSON object."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or None if `path` doesn't exist. A malformed file raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)
