"""Generate an LLM-friendly knowledge index from the DeepLabCut documentation.

The index is `knowledge/<version>/{api,docs}.jsonl`, one JSON object per line,
plus `llms.txt` at the site root. See README.md for the layout and the CLI.
"""

from __future__ import annotations

from .api_index import ApiNode, Symbol, build_api_nodes
from .docs_index import DocsPageNode, Section, build_docs_nodes
from .llms_txt import build_llms_txt
from .schemas import ApiRecord, DocPageRecord, DocSectionRecord, TopManifest, VersionManifest
from .toc import TocEntry, read_toc
from .write import write_top_manifest, write_version

__all__ = [
    "ApiNode",
    "ApiRecord",
    "DocPageRecord",
    "DocSectionRecord",
    "DocsPageNode",
    "Section",
    "Symbol",
    "TocEntry",
    "TopManifest",
    "VersionManifest",
    "build_api_nodes",
    "build_docs_nodes",
    "build_llms_txt",
    "read_toc",
    "write_top_manifest",
    "write_version",
]
