"""Build `llms.txt`, the spec-format entry point at the docs site root.

https://llmstxt.org -- an H1 title, a one-line blockquote description, then
`##` sections of markdown links. Only top-level pages are listed per part
(nested pages are reachable through a page's own `children` in the structured
index); this file is meant to orient an agent, not enumerate every page --
that is what `knowledge/manifest.json` and `docs.jsonl` are for.
"""

from __future__ import annotations

from collections.abc import Sequence

from .docs_index import DocsPageNode

PROJECT_NAME = "DeepLabCut"
PROJECT_SUMMARY = "Markerless pose-estimation of user-defined features with deep learning"

REPO_URL = "https://github.com/DeepLabCut/DeepLabCut"
ISSUES_URL = f"{REPO_URL}/issues"
FORUM_URL = "https://forum.image.sc/tag/deeplabcut"


def build_llms_txt(
    docs_pages: Sequence[DocsPageNode],
    api_base_url: str,
    knowledge_base_url: str,
    version: str,
) -> str:
    """Render `llms.txt` for the docs build at `docs_pages` / `api_base_url`.

    `knowledge_base_url` is the published `.../knowledge/` directory; `version`
    is the one whose `api.jsonl` and `docs.jsonl` sit directly under it (the
    same version this whole build is for).
    """
    lines = [
        f"# {PROJECT_NAME}",
        "",
        f"> {PROJECT_SUMMARY}",
        "",
    ]

    for part, pages in _parts(docs_pages):
        lines.append(f"## {part}")
        for page in pages:
            summary = f": {page.summary}" if page.summary else ""
            lines.append(f"- [{page.title}]({page.docs_url}){summary}")
        lines.append("")

    lines += [
        "## API reference",
        f"- [Full API reference]({api_base_url}reference/): every public module, "
        "class and function, generated from docstrings",
        "",
        "## Machine-readable index",
        f"- [Index manifest]({knowledge_base_url}manifest.json): lists every "
        "indexed api version and which one is current",
        f"- [Doc pages and sections]({knowledge_base_url}{version}/docs.jsonl): "
        "one JSON object per line, `{id, title, url, section, summary, ...}`",
        f"- [API symbols]({knowledge_base_url}{version}/api.jsonl): one JSON "
        "object per line, `{id, kind, name, module, signature, summary, ...}`",
        "",
        "## Optional",
        f"- [Source code]({REPO_URL}): the DeepLabCut repository",
        f"- [Issue tracker]({ISSUES_URL}): report bugs, request features",
        f"- [Community forum]({FORUM_URL}): usage questions and support",
        "",
    ]

    return "\n".join(lines)


def _parts(docs_pages: Sequence[DocsPageNode]) -> list[tuple[str, list[DocsPageNode]]]:
    """Top-level pages (no parent), grouped by `part` in first-seen order."""
    grouped: dict[str, list[DocsPageNode]] = {}
    for page in docs_pages:
        if page.parent:
            continue
        grouped.setdefault(page.part or "Docs", []).append(page)
    return list(grouped.items())
