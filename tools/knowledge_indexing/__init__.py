"""Generate an LLM-friendly knowledge index from the DeepLabCut documentation.

The index is `knowledge/<version>/{api,docs}.jsonl`, one JSON object per line,
plus `llms.txt` at the site root. See README.md for the layout and the CLI.

Only what `griffe`/`docutils`/`markdown-it-py`-free code needs.
`api_index` and `docs_index` are not re-exported here.
"""

from __future__ import annotations

from .llms_txt import build_llms_txt
from .schemas import ApiRecord, DocPageRecord, DocSectionRecord, TopManifest, VersionManifest
from .toc import TocEntry, read_toc
from .write import delete_version, write_top_manifest, write_version

__all__ = [
    "ApiRecord",
    "DocPageRecord",
    "DocSectionRecord",
    "TocEntry",
    "TopManifest",
    "VersionManifest",
    "build_llms_txt",
    "delete_version",
    "read_toc",
    "write_top_manifest",
    "write_version",
]
