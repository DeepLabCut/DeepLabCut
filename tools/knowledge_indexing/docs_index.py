"""Extract docs-page nodes from the user documentation.

`_toc.yml` defines the scope: it lists the published pages and nests them, so it
supplies each node's place in the hierarchy. Pages are read from their markdown
source, so no built documentation is required.

Each heading below the page title becomes a section: a retrievable part of one
page, addressed by the anchor Sphinx publishes for it and carrying the first
paragraph of prose beneath it.

Markdown is parsed with markdown-it-py, the parser Jupyter Book uses. The docs
nest code fences up to five backticks deep, and the contents of a fence must not
be read as markdown.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from docutils.nodes import make_id
from markdown_it import MarkdownIt
from markdown_it.token import Token

from .schemas import DOCS_NAMESPACE, DocsPageNode, Section
from .toc import TOC_FILE, TocEntry, read_toc

# Ids are relative to this directory, so `docs/installation` becomes
# `docs:installation` while root pages such as `README` keep their name.
DOCS_DIR = "docs/"

# Excerpts are trimmed to this length at a word boundary.
EXCERPT_MAX_CHARS = 300

# Suffixes that mark a link as pointing at another source file rather than at a
# MyST label.
FILE_SUFFIXES = (".md", ".ipynb", ".html")

# A MyST target definition on its own line, e.g. `(sec:uv-install)=`.
MYST_LABEL = re.compile(r"^\(([^)\s]+)\)=[ \t]*$", re.MULTILINE)

# The MyST cross-reference role, e.g. {ref}`GPU support <sec:install-gpu-support>`
# or {ref}`sec:install-gpu-support`. Captures the target, not the link text.
MYST_REF = re.compile(r"\{ref\}`(?:[^`<]*<)?([^`<>]+?)>?`")

# Paragraphs that are markup rather than prose: MyST directive fences, target
# definitions, notebook magics and raw HTML.
MARKUP_PARAGRAPH = re.compile(r"^(:::|%%|<)|^\(.+\)=$")

# MyST directives whose body is prose. Others, such as `{figure}` or
# `{tab-set}`, hold a caption or nested layout instead.
PROSE_DIRECTIVES = frozenset(
    {
        "admonition",
        "attention",
        "caution",
        "danger",
        "error",
        "hint",
        "important",
        "note",
        "seealso",
        "tip",
        "warning",
    }
)
DIRECTIVE_NAME = re.compile(r"^\{([a-z-]+)\}")

# Options at the top of a directive body, either a `---` delimited block or a
# field list, e.g. `:class: dropdown`.
DIRECTIVE_OPTION = re.compile(r"^\s*:[\w-]+:")

# A MyST role with an explicit label, as it survives inline extraction:
# "{ref}GPU support <sec:install-gpu-support>". Captures the label.
MYST_ROLE_LABEL = re.compile(r"\{[a-z-]+\}([^<\n]*?)\s*<[^>\n]+>")
MYST_ROLE = re.compile(r"\{[a-z-]+\}")

_MARKDOWN = MarkdownIt("commonmark")


@dataclass(frozen=True)
class ParsedPage:
    """A markdown page, parsed but not yet resolved against the other pages."""

    entry: TocEntry
    local_id: str
    title: str
    summary: str = ""
    sections: tuple[Section, ...] = ()
    labels: tuple[str, ...] = ()
    link_targets: tuple[str, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)


def build_docs_nodes(repo: Path, base_url: str = "") -> list[DocsPageNode]:
    """Build one node per published markdown page listed in `_toc.yml`.

    `base_url` is prepended to every published URL.
    """
    pages = []
    for entry in read_toc(repo / TOC_FILE):
        path = repo / f"{entry.file}.md"
        # Notebook entries have no markdown source.
        if path.is_file():
            page = _parse_page(path, entry, base_url)
            if page is not None:
                pages.append(page)

    local_ids = {page.local_id for page in pages}
    labels = {alias: page.local_id for page in pages for alias in _label_aliases(page.labels)}
    return [_to_node(page, labels, local_ids, base_url) for page in pages]


def _parse_page(path: Path, entry: TocEntry, base_url: str) -> ParsedPage | None:
    """Parse one page, or return None if its frontmatter asks to be ignored."""
    frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    audit = frontmatter.get("deeplabcut") or {}
    if audit.get("ignore"):
        return None

    local_id = _local_id(entry.file)
    tokens = _MARKDOWN.parse(body)
    title, summary, sections = _read_structure(tokens, f"{DOCS_NAMESPACE}:{local_id}", _page_url(entry, base_url))

    return ParsedPage(
        entry=entry,
        local_id=local_id,
        title=title or local_id,
        summary=summary,
        sections=sections,
        labels=_unique(MYST_LABEL.findall(body)),
        link_targets=_unique([*_link_targets(tokens), *MYST_REF.findall(body)]),
        audit=audit,
    )


def _to_node(page: ParsedPage, labels: dict[str, str], local_ids: set[str], base_url: str) -> DocsPageNode:
    """Turn a parsed page into a node, resolving its links to page ids.

    `_toc.yml` also nests notebooks, which have no node, so `parent` and
    `children` are filtered against the pages that were indexed.
    """
    entry = page.entry
    parent = _local_id(entry.parent)
    return DocsPageNode(
        id=f"{DOCS_NAMESPACE}:{page.local_id}",
        title=page.title,
        docs_url=_page_url(entry, base_url),
        source_file=f"{entry.file}.md",
        part=entry.part,
        parent=_qualify(parent) if parent in local_ids else "",
        children=tuple(_qualify(child) for child in map(_local_id, entry.children) if child in local_ids),
        summary=page.summary,
        status=str(page.audit.get("status") or ""),
        visibility=str(page.audit.get("visibility") or ""),
        last_verified=str(page.audit.get("last_verified") or ""),
        sections=page.sections,
        related_pages=_related_pages(page, labels, local_ids),
        labels=page.labels,
    )


def _page_url(entry: TocEntry, base_url: str) -> str:
    """Published URL of a page. Jupyter Book mirrors the source layout as HTML."""
    return f"{base_url}{entry.file}.html"


def _read_structure(tokens: list[Token], page_id: str, docs_url: str) -> tuple[str, str, tuple[Section, ...]]:
    """Page title, page summary, and one section per heading below the title."""
    starts = [index for index, token in enumerate(tokens) if token.type == "heading_open"]
    bounds = [*starts, len(tokens)]

    title = ""
    summary = _first_paragraph(tokens[: bounds[0]]) if starts else ""
    sections = []
    seen: Counter[str] = Counter()

    for start, end in zip(bounds, bounds[1:], strict=False):
        heading = tokens[start]
        text = _inline_text(tokens[start + 1])
        excerpt = _first_paragraph(tokens[start + 2 : end])

        if heading.tag == "h1":
            # The first H1 titles the page; its lead paragraph summarises it.
            title = title or text
            summary = summary or excerpt
            continue

        anchor = _anchor(text)
        seen[anchor] += 1
        # Sphinx registers only the first occurrence of a repeated heading, so
        # the anchor is shared and the id is suffixed to stay unique.
        suffix = "" if seen[anchor] == 1 else f"-{seen[anchor]}"
        sections.append(
            Section(
                id=f"{page_id}#{anchor}{suffix}",
                title=text,
                level=int(heading.tag[1]),
                anchor=anchor,
                docs_url=f"{docs_url}#{anchor}",
                excerpt=excerpt,
            )
        )

    return title, summary, tuple(sections)


def _first_paragraph(tokens: list[Token], depth: int = 0) -> str:
    """First paragraph of prose in `tokens`, trimmed to one short line.

    Paragraphs holding only an image or a MyST target are skipped, and so is
    fenced code — except for admonition directives, whose body is prose and is
    parsed in turn.
    """
    for index, token in enumerate(tokens):
        if token.type == "paragraph_open":
            text = _inline_text(tokens[index + 1])
            if text and not MARKUP_PARAGRAPH.match(text):
                return _shorten(text)
        elif token.type == "fence" and depth < 2:
            text = _directive_prose(token, depth)
            if text:
                return text
    return ""


def _directive_prose(fence: Token, depth: int) -> str:
    """First paragraph inside an admonition directive, or "" for other fences."""
    name = DIRECTIVE_NAME.match(fence.info.strip())
    if name is None or name.group(1) not in PROSE_DIRECTIVES:
        return ""
    return _first_paragraph(_MARKDOWN.parse(_directive_body(fence.content)), depth + 1)


def _directive_body(content: str) -> str:
    """Directive content with its leading options stripped."""
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        closing = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), 0)
        lines = lines[closing + 1 :]
    while lines and DIRECTIVE_OPTION.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines)


def _inline_text(token: Token) -> str:
    """Rendered plain text of an inline token.

    Only text and inline code are kept, dropping link and emphasis markup so the
    result matches what a reader sees.
    """
    if token.children is None:
        return _strip_roles(token.content)
    # Line breaks carry no content of their own and become spaces, or the words
    # on either side would run together.
    text = "".join(
        " " if child.type in ("softbreak", "hardbreak") else child.content
        for child in token.children
        if child.type in ("text", "code_inline", "softbreak", "hardbreak")
    )
    return _strip_roles(text)


def _strip_roles(text: str) -> str:
    """Reduce MyST roles to their label.

    CommonMark has no notion of roles, so `` {ref}`GPU support <sec:gpu>` ``
    arrives here as "{ref}GPU support <sec:gpu>", while Sphinx renders only the
    label.
    """
    text = MYST_ROLE_LABEL.sub(lambda match: match.group(1), text)
    return MYST_ROLE.sub("", text).strip()


def _link_targets(tokens: list[Token]) -> list[str]:
    """Every link target in the document, in order."""
    return [
        child.attrGet("href") or ""
        for token in tokens
        if token.type == "inline"
        for child in token.children or ()
        if child.type == "link_open"
    ]


def _anchor(text: str) -> str:
    """Heading anchor, as Sphinx publishes it.

    `make_id` is what docutils uses to turn a heading into a section id: it drops
    non-ASCII, collapses everything else to hyphens and trims leading digits, so
    "(1) Create a New 3D Project:" anchors as `create-a-new-3d-project` and
    "DeepLabCut's role" as `deeplabcuts-role`.
    """
    return make_id(text)


def _shorten(text: str, limit: int = EXCERPT_MAX_CHARS) -> str:
    """Collapse whitespace and trim to `limit` at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the markdown body."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}, parts[2]
    return (frontmatter if isinstance(frontmatter, dict) else {}), parts[2]


def _local_id(toc_file: str) -> str:
    """Page id without its namespace, e.g. `docs/pytorch/user_guide` -> `pytorch/user_guide`."""
    return toc_file.removeprefix(DOCS_DIR)


def _qualify(local_id: str) -> str:
    return f"{DOCS_NAMESPACE}:{local_id}"


def _label_aliases(labels: tuple[str, ...]) -> list[str]:
    """Every spelling a label can be referenced by.

    MyST labels are often namespaced, e.g. `(file:how-to-install)=`, but pages
    link to them either with or without the prefix.
    """
    aliases = []
    for label in labels:
        aliases.append(label)
        if ":" in label:
            aliases.append(label.split(":", 1)[1])
    return aliases


def _related_pages(page: ParsedPage, labels: dict[str, str], local_ids: set[str]) -> tuple[str, ...]:
    """Ids of the other pages this page links to."""
    directory = posixpath.dirname(page.entry.file)
    resolved = (_resolve_target(target, directory, labels) for target in page.link_targets)
    return _unique(_qualify(local_id) for local_id in resolved if local_id in local_ids and local_id != page.local_id)


def _resolve_target(target: str, directory: str, labels: dict[str, str]) -> str:
    """Resolve a link or cross-reference target to a page id without namespace.

    Returns "" for external links, in-page anchors, assets, and targets that
    cannot be matched to a page.
    """
    target = target.split("#", 1)[0].strip()
    if not target or target.startswith(("http://", "https://", "mailto:", "/")):
        return ""

    if PurePosixPath(target).suffix not in FILE_SUFFIXES:
        return labels.get(target, "")

    resolved = posixpath.normpath(posixpath.join(directory, target)).strip("/")
    if resolved.startswith(".."):
        return ""
    return _local_id(re.sub(r"\.(md|ipynb|html)$", "", resolved))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate while preserving order, dropping empties."""
    return tuple(dict.fromkeys(value for value in values if value))
