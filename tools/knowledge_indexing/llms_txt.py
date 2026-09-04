"""Build `llms.txt`, the spec-format entry point at the docs site root.

https://llmstxt.org -- an H1 title, a one-line blockquote description, then
`##` sections of markdown links. Points at the structured knowledge index
rather than listing pages itself: `docs.jsonl`/`api.jsonl` already cover every
page and section with better fidelity, so restating them here would just be a
second, staler copy.
"""

from __future__ import annotations

PROJECT_NAME = "DeepLabCut"
PROJECT_SUMMARY = "Markerless pose-estimation of user-defined features with deep learning"

REPO_URL = "https://github.com/DeepLabCut/DeepLabCut"
ISSUES_URL = f"{REPO_URL}/issues"
FORUM_URL = "https://forum.image.sc/tag/deeplabcut"


def build_llms_txt(docs_base_url: str, api_base_url: str, knowledge_base_url: str, version_label: str) -> str:
    """Render `llms.txt`. `knowledge_base_url` is the published `.../knowledge/` directory."""
    lines = [
        f"# {PROJECT_NAME}",
        "",
        f"> {PROJECT_SUMMARY}",
        "",
        "## Docs",
        f"- [User documentation]({docs_base_url}): full user guide and workflows",
        f"- [API reference]({api_base_url}reference/): every public module, class and function",
        "",
        "## Machine-readable index",
        f"- [Index manifest]({knowledge_base_url}manifest.json): start here -- every indexed "
        "api version and which one is current",
        f"- [Doc pages and sections]({knowledge_base_url}{version_label}/docs.jsonl): one JSON "
        "object per line, `{id, title, url, section, summary, ...}`",
        f"- [API symbols]({knowledge_base_url}{version_label}/api.jsonl): one JSON object per "
        "line, `{id, kind, name, module, signature, summary, ...}`",
        "",
        "## Optional",
        f"- [Source code]({REPO_URL}): the DeepLabCut repository",
        f"- [Issue tracker]({ISSUES_URL}): report bugs, request features",
        f"- [Community forum]({FORUM_URL}): usage questions and support",
        "",
    ]
    return "\n".join(lines)
