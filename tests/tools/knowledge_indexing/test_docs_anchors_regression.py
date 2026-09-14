"""Regression check: every anchor we publish exists in the real docs build.

The unit tests check `docs_index` against a hand-written HTML fixture; this
checks that fixture against the toolchain, by building a page with the repo's
own `_config.yml`. A docs upgrade that changes how headings become ids fails
here rather than silently shipping links to the wrong section.

Marked `functional` and skipped without jupyter-book, which lives in the `docs`
extra rather than the `dev` group.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from html import unescape
from pathlib import Path

import pytest
import yaml
from markdown_it import MarkdownIt

from tools.knowledge_indexing.docs_index import _read_structure, read_published_anchors

pytestmark = [
    pytest.mark.functional,
    pytest.mark.skipif(shutil.which("jupyter-book") is None, reason="jupyter-book (the `docs` extra) not installed"),
]

PAGE = """\
# Sample page

Lead paragraph.

## Overview

First.

## Code example

Second.

## Overview

Third, a repeat.

## Code example:

Fourth, colliding only through punctuation.

(sec:labelled-section)=
## Labelled section

Fifth, carrying an explicit MyST target.

## Output & directory structure

Sixth, with an entity in the heading.

## Overview

Seventh, a third repeat.
"""


def _build(tmp_path: Path) -> str:
    """Build PAGE with the repo's config and return the generated HTML."""
    repo = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((repo / "_config.yml").read_text(encoding="utf-8"))
    # These point at files under docs/ that this minimal book does not carry.
    config.pop("logo", None)
    config.get("sphinx", {}).get("config", {}).pop("html_static_path", None)
    config.get("sphinx", {}).get("config", {}).pop("html_css_files", None)

    book = tmp_path / "book"
    book.mkdir()
    (book / "_config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (book / "_toc.yml").write_text("format: jb-book\nroot: page\n", encoding="utf-8")
    (book / "page.md").write_text(PAGE, encoding="utf-8")

    result = subprocess.run(
        [shutil.which("jupyter-book"), "build", str(book)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    html = book / "_build" / "html" / "page.html"
    if not html.is_file():
        pytest.fail(f"jupyter-book produced no page.html\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return html.read_text(encoding="utf-8")


def _indexed(html: str):
    tokens = MarkdownIt("commonmark").parse(PAGE)
    anchors = read_published_anchors(html)
    _, _, sections = _read_structure(tokens, "docs:page", "https://example.test/page.html", anchors)
    return sections


_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>", re.S)


def _heading_at(page: str, anchor: str) -> str:
    """Text of the heading a fragment lands on, read independently of `docs_index`."""
    start = page.find(f'id="{anchor}"')
    assert start != -1, f"#{anchor} is not on the page"
    match = _HEADING.search(page, start)
    assert match, f"no heading follows #{anchor}"
    return " ".join(unescape(_TAG.sub("", match.group(2))).split()).rstrip("#").strip()


def test_every_anchor_lands_on_its_own_heading(tmp_path: Path):
    html = _build(tmp_path)
    sections = _indexed(html)

    assert sections, "no sections were indexed"
    for section in sections:
        assert section.anchor, f"{section.title!r} got no anchor from a page that was built"
        assert _heading_at(html, section.anchor) == section.title


def test_repeated_headings_resolve_to_distinct_anchors(tmp_path: Path):
    html = _build(tmp_path)
    overviews = [s for s in _indexed(html) if s.title == "Overview"]

    assert len(overviews) == 3
    anchors = [s.anchor for s in overviews]
    # All three would otherwise publish #overview, deep-linking to the first.
    assert len(set(anchors)) == 3, f"repeated headings share an anchor: {anchors}"
    assert anchors[0] == "overview"


def test_headings_needing_more_than_a_slug_are_resolved(tmp_path: Path):
    html = _build(tmp_path)
    by_title = {s.title: s for s in _indexed(html)}

    # An entity in the heading must not leave the section unanchored.
    assert by_title["Output & directory structure"].anchor == "output-directory-structure"
    # make_id strips the colon, so this collides and cannot keep that slug.
    assert by_title["Code example:"].anchor != by_title["Code example"].anchor
    # The MyST target case: whichever id the build chose, it must be the one
    # that lands on this heading and not merely an id present on the page.
    assert _heading_at(html, by_title["Labelled section"].anchor) == "Labelled section"
