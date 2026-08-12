"""Generate an LLM-friendly knowledge index from the DeepLabCut documentation.

The index is a directory of small YAML files, one per node, that an agent can
load lazily by id. See README.md for the layout and the CLI.
"""

from __future__ import annotations

from .api_index import build_api_nodes
from .docs_index import build_docs_nodes
from .schemas import (
    ApiNode,
    DocsPageNode,
    Manifest,
    Node,
    Section,
    Symbol,
    SymbolEntry,
    SymbolTable,
)
from .toc import TocEntry, read_toc
from .write import write_index, write_symbol_table

__all__ = [
    "ApiNode",
    "DocsPageNode",
    "Manifest",
    "Node",
    "Section",
    "Symbol",
    "SymbolEntry",
    "SymbolTable",
    "TocEntry",
    "build_api_nodes",
    "build_docs_nodes",
    "read_toc",
    "write_index",
    "write_symbol_table",
]
