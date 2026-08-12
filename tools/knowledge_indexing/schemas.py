"""On-disk schema of the knowledge index.

Everything the index publishes is defined here: the two node types, the nested
structures they contain, and the two top-level files, `index.yaml` and
`symbols.yaml`. Each is a frozen dataclass that renders itself as a plain dict, so
`write.py` only has to serialise what it is handed.

Structures used while *reading* inputs are not schema and live with the code that
reads them: `TocEntry` in `toc.py` describes an entry of `_toc.yml`, and
`ParsedPage` in `docs_index.py` is a page held between its two parsing passes.
Neither is ever written out.

Ids are namespaced by node type (`docs:`, `api:`), which makes a reference
unambiguous about what it points at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

# Bumped when a published file changes shape, so a consumer can refuse an index
# it does not understand. Recorded in `index.yaml` and `symbols.yaml`.
SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "0.2.0"
GENERATED_BY = "tools/knowledge_indexing"

NAMESPACE_SEPARATOR = ":"
DOCS_NAMESPACE = "docs"
API_NAMESPACE = "api"

# Layout of an index directory.
API_DIRECTORY = "apis"
DOCS_DIRECTORY = "docs-pages"
MANIFEST = "index.yaml"
SYMBOL_TABLE = "symbols.yaml"


class Node:
    """Base class for index nodes.

    Subclasses are dataclasses with an `id` field; they set `node_type` and
    implement `to_dict`.
    """

    node_type: ClassVar[str]
    id: str

    @property
    def filename(self) -> str:
        """YAML filename for this node.

        The namespace prefix is dropped and slashes are flattened, since page ids
        nest (`docs:pytorch/user_guide`) while node directories do not.
        """
        local = self.id.split(NAMESPACE_SEPARATOR, 1)[-1]
        return f"{local.replace('/', '--')}.yaml"

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol:
    """A documented function or class, as published on an API reference page."""

    name: str
    kind: str
    summary: str
    signature: str
    source: str
    docs_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "signature": self.signature,
            "summary": self.summary,
            "source": self.source,
            "docs_url": self.docs_url,
        }


@dataclass(frozen=True)
class Section:
    """A heading within a docs page, retrievable in its own right.

    `anchor` addresses it on the published page; `excerpt` is the first paragraph
    of prose beneath the heading.
    """

    id: str
    title: str
    level: int
    anchor: str
    docs_url: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "anchor": self.anchor,
            "docs_url": self.docs_url,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class ApiNode(Node):
    """One published module and the symbols documented on its reference page."""

    node_type: ClassVar[str] = API_NAMESPACE

    id: str
    module: str
    summary: str
    source: str
    docs_url: str
    symbols: tuple[Symbol, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type,
            "title": self.module,
            "module": self.module,
            "summary": self.summary,
            "source": self.source,
            "docs_url": self.docs_url,
            "symbols": {symbol.name: symbol.to_dict() for symbol in self.symbols},
        }


@dataclass(frozen=True)
class DocsPageNode(Node):
    """One page of the user documentation.

    `part`, `parent` and `children` come from `_toc.yml` and place the page in the
    published navigation. `status`, `visibility` and `last_verified` are copied
    from the page's audit frontmatter; nothing is filtered on them. `labels` are
    the MyST targets the page defines, which is what a `{ref}` elsewhere in the
    docs resolves against.
    """

    node_type: ClassVar[str] = "docs-page"

    id: str
    title: str
    docs_url: str
    source_file: str
    part: str = ""
    parent: str = ""
    children: tuple[str, ...] = ()
    summary: str = ""
    status: str = ""
    visibility: str = ""
    last_verified: str = ""
    sections: tuple[Section, ...] = ()
    related_pages: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type,
            "title": self.title,
            "docs_url": self.docs_url,
            "source_file": self.source_file,
            "part": self.part or None,
            "parent": self.parent or None,
            "children": list(self.children),
            "status": self.status or None,
            "visibility": self.visibility or None,
            "last_verified": self.last_verified or None,
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections],
            "related_pages": list(self.related_pages),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class SymbolEntry:
    """One row of the symbol table: where a symbol is documented."""

    kind: str
    module: str
    file: str
    docs_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "module": self.module,
            "file": self.file,
            "docs_url": self.docs_url,
        }


@dataclass(frozen=True)
class SymbolTable:
    """`symbols.yaml`: symbol id -> the module node documenting it.

    API nodes are grouped per module, so this is what resolves a bare symbol name
    to a node.
    """

    symbols: dict[str, SymbolEntry] = field(default_factory=dict)

    @classmethod
    def from_apis(cls, apis: Sequence[ApiNode]) -> SymbolTable:
        return cls(
            {
                f"{node.id}.{symbol.name}": SymbolEntry(
                    kind=symbol.kind,
                    module=node.id,
                    file=f"{API_DIRECTORY}/{node.filename}",
                    docs_url=symbol.docs_url,
                )
                for node in apis
                for symbol in node.symbols
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "symbols": {name: entry.to_dict() for name, entry in self.symbols.items()},
        }


@dataclass(frozen=True)
class Manifest:
    """`index.yaml`: what the index holds, and which file holds each node.

    Every node id is listed with its file, because ids are namespaced and nested
    while filenames are neither, so the mapping cannot be derived from an id.

    `base_urls` names the sites the published URLs were built against. Those URLs
    are absolute in the nodes themselves, so this is what a consumer would rebase
    onto a local build.
    """

    version: str
    generated_at: str
    nodes: dict[str, Sequence[Node]] = field(default_factory=dict)
    revision: str = ""
    base_urls: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "revision": self.revision or None,
            "extractor_version": EXTRACTOR_VERSION,
            "generated_by": GENERATED_BY,
            "generated_at": self.generated_at,
            "base_urls": self.base_urls,
            "node_counts": {directory: len(nodes) for directory, nodes in self.nodes.items()},
            "nodes": {
                directory: [{"id": node.id, "file": f"{directory}/{node.filename}"} for node in nodes]
                for directory, nodes in self.nodes.items()
            },
        }
