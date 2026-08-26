"""On-disk schema of the knowledge index.

Everything the index publishes is defined here: the JSONL record shapes
written to `docs.jsonl` and `api.jsonl`, and the two manifest shapes --
`manifest.json` (one per version) and the top-level `knowledge/manifest.json`
that enumerates every version. Each is a frozen dataclass that renders itself
as a plain dict, so `write.py` only has to serialise what it is handed.

Structures used while *reading* inputs are not schema and live with the code
that reads them: `TocEntry` in `toc.py` describes an entry of `_toc.yml`,
`ParsedPage` in `docs_index.py` is a page held between its two parsing
passes, and `ApiNode` / `Symbol` (api_index.py) and `DocsPageNode` / `Section`
(docs_index.py) are the build-time trees `write.py` flattens into the records
below. None of them is ever written out directly.

Ids are namespaced by node type (`docs:`, `api:`), which makes a reference
unambiguous about what it points at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Bumped when a published file changes shape, so a consumer can refuse an
# index it does not understand. Recorded in every manifest.json.
SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "0.0.1"
GENERATED_BY = "tools/knowledge_indexing"

NAMESPACE_SEPARATOR = ":"
DOCS_NAMESPACE = "docs"
API_NAMESPACE = "api"

# Layout under the output directory: <output>/knowledge/<version>/{api,docs}.jsonl
# plus <output>/llms.txt. See README.md.
KNOWLEDGE_DIR = "knowledge"
DOCS_FILE = "docs.jsonl"
API_FILE = "api.jsonl"
VERSION_MANIFEST = "manifest.json"
TOP_MANIFEST = "manifest.json"
LLMS_TXT = "llms.txt"


@dataclass(frozen=True)
class DocPageRecord:
    """One line of `docs.jsonl`: a published user-docs page.

    Carries the page's place in the docs (`section`, `parent`, `children`) so
    a `DocSectionRecord` on it does not have to repeat any of it.
    """

    id: str
    title: str
    url: str
    source_file: str
    section: str = ""
    summary: str = ""
    status: str = ""
    last_verified: str = ""
    parent: str = ""
    children: tuple[str, ...] = ()
    related_pages: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    type: str = "page"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "url": self.url,
            "section": self.section or None,
            "summary": self.summary,
            "status": self.status or None,
            "last_verified": self.last_verified or None,
            "source_file": self.source_file,
            "parent": self.parent or None,
            "children": list(self.children),
            "related_pages": list(self.related_pages),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class DocSectionRecord:
    """One line of `docs.jsonl`: a heading within a page, retrievable on its own.

    `page` is the id of the `DocPageRecord` it belongs to; `section` is
    denormalised from that page so this row still stands on its own without a
    join back to it.
    """

    id: str
    page: str
    title: str
    url: str
    anchor: str = ""
    level: int = 0
    section: str = ""
    summary: str = ""
    type: str = "section"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "page": self.page,
            "title": self.title,
            "level": self.level,
            "anchor": self.anchor,
            "url": self.url,
            "section": self.section or None,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ApiRecord:
    """One line of `api.jsonl`: a module or one documented symbol on it.

    `kind` is `"module"` for a module's own docstring, or `"function"` /
    `"class"` for a symbol it publishes. `signature` is empty for modules.
    """

    id: str
    kind: str
    name: str
    module: str
    url: str
    signature: str = ""
    summary: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "module": self.module,
            "signature": self.signature,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
        }


@dataclass(frozen=True)
class VersionManifest:
    """`knowledge/<version>/manifest.json`: provenance of one version's build."""

    version: str
    revision: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "revision": self.revision or None,
            "extractor_version": EXTRACTOR_VERSION,
            "generated_by": GENERATED_BY,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class TopManifest:
    """`knowledge/manifest.json`: the single enumeration point for the index.

    gh-pages gives no directory listing, so this is how a consumer discovers
    which api versions exist and which one is current. `docs_path` is "" for
    an index with no unversioned docs build yet.
    """

    docs_path: str
    api_latest: str
    api_versions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "docs": {"versioned": False, "path": self.docs_path or None},
            "api": {
                "versioned": True,
                "latest": self.api_latest,
                "versions": list(self.api_versions),
            },
        }
