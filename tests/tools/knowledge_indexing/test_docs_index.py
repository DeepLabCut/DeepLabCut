from __future__ import annotations

from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from tools.knowledge_indexing.docs_index import (
    _page_anchors,
    _read_structure,
    _split_frontmatter,
    read_published_anchors,
)
from tools.knowledge_indexing.toc import TocEntry

PAGE_URL = "https://example.test/docs/page.html"


def _sections(markdown: str, anchors: dict[tuple[str, int], str] | None = None):
    tokens = MarkdownIt("commonmark").parse(markdown)
    _, _, sections = _read_structure(tokens, "docs:page", PAGE_URL, anchors)
    return sections


# A built page as Sphinx emits it: repeated headings get auto ids, and each
# heading carries a "#" permalink that is not part of its text.
BUILT_HTML = """
<section id="overview">
<h2>Overview<a class="headerlink" href="#overview" title="Link">#</a></h2>
<p>First.</p>
<section id="output-directory-structure">
<h3>Output &amp; directory structure<a class="headerlink" href="#output-directory-structure">#</a></h3>
</section>
</section>
<section id="id1">
<h2>Overview<a class="headerlink" href="#id1" title="Link">#</a></h2>
<p>Second.</p>
</section>
"""


def test_read_published_anchors_keys_by_text_and_occurrence():
    anchors = read_published_anchors(BUILT_HTML)
    assert anchors[("Overview", 1)] == "overview"
    assert anchors[("Overview", 2)] == "id1"


def test_read_published_anchors_strips_the_permalink_and_resolves_entities():
    anchors = read_published_anchors(BUILT_HTML)
    # Both have to match the markdown heading text for the lookup to hit.
    assert anchors[("Output & directory structure", 1)] == "output-directory-structure"


def test_sections_use_the_published_anchor():
    markdown = "# Title\n\n## Overview\n\nFirst.\n\n## Overview\n\nSecond.\n"
    first, second = _sections(markdown, read_published_anchors(BUILT_HTML))

    assert first.anchor == "overview"
    assert first.docs_url == f"{PAGE_URL}#overview"
    # The repeat gets its own published id, not the first heading's anchor.
    assert second.anchor == "id1"
    assert second.docs_url == f"{PAGE_URL}#id1"


def test_sections_fall_back_to_the_page_without_a_build():
    first, second = _sections("# Title\n\n## Overview\n\nA.\n\n## Overview\n\nB.\n")

    # No anchors available: link to the page rather than guess a fragment.
    for section in (first, second):
        assert section.anchor == ""
        assert section.docs_url == PAGE_URL


def test_heading_the_build_publishes_no_section_for_gets_no_anchor():
    markdown = "# Title\n\n## Overview\n\nA.\n\n## Not A Section\n\nB.\n"
    _, orphan = _sections(markdown, read_published_anchors(BUILT_HTML))

    assert orphan.title == "Not A Section"
    assert orphan.anchor == ""
    assert orphan.docs_url == PAGE_URL


def test_record_ids_are_stable_with_and_without_a_build():
    markdown = "# Title\n\n## Overview\n\nA.\n\n## Overview\n\nB.\n"
    # Identity comes from the source, so it does not depend on the build.
    with_build = [s.id for s in _sections(markdown, read_published_anchors(BUILT_HTML))]
    without_build = [s.id for s in _sections(markdown)]

    assert with_build == without_build == ["docs:page#overview", "docs:page#overview~2"]


def test_page_title_h1_does_not_consume_a_later_heading_s_occurrence():
    # The build counts the title when numbering repeats, so the lookup must too;
    # otherwise the h2 asks for ("Overview", 1) and gets the title's anchor.
    html = '<section id="overview"><h2>Overview</h2></section>'
    anchors = read_published_anchors('<section id="intro"><h1>Overview</h1></section>' + html)

    (section,) = _sections("# Overview\n\nLead.\n\n## Overview\n\nBody.\n", anchors)
    assert section.anchor == "overview"


def test_only_the_first_h1_is_the_page_title():
    # Some pages use h1 throughout, so only the first may be taken as the title.
    sections = _sections("# Title\n\nLead.\n\n# Second\n\nA.\n\n# Third\n\nB.\n")

    assert [s.title for s in sections] == ["Second", "Third"]
    assert [s.level for s in sections] == [1, 1]


def test_heading_that_slugs_to_nothing_still_gets_a_usable_id():
    # make_id strips these entirely, leaving the record named "docs:page#".
    first, second = _sections("# Title\n\n## ???\n\nA.\n\n## !!!\n\nB.\n")

    for section in (first, second):
        assert section.id.startswith("docs:page#section-")
    assert first.id != second.id


def test_page_anchors_reads_the_html_mirroring_the_toc_path(tmp_path: Path):
    # Pins the layout production expects of a downloaded docs artifact:
    # <html_dir>/<toc entry>.html, source layout mirrored.
    page = tmp_path / "docs" / "main-workflows" / "user-guide.html"
    page.parent.mkdir(parents=True)
    page.write_text(BUILT_HTML, encoding="utf-8")

    entry = TocEntry(file="docs/main-workflows/user-guide")
    assert _page_anchors(tmp_path, entry)[("Overview", 1)] == "overview"


def test_page_anchors_raises_when_a_requested_page_is_not_built(tmp_path: Path):
    # Asking for anchors and silently getting page-level links back is the
    # failure this design exists to prevent, so a missing page must not degrade.
    with pytest.raises(FileNotFoundError, match="every page in"):
        _page_anchors(tmp_path, TocEntry(file="docs/missing"))


def test_page_anchors_without_a_build_is_not_an_error():
    assert _page_anchors(None, TocEntry(file="docs/missing")) == {}


def test_frontmatter_is_split_from_body():
    frontmatter, body = _split_frontmatter("---\nstatus: verified\n---\n# Title\n\nProse.\n")
    assert frontmatter == {"status": "verified"}
    assert body == "# Title\n\nProse.\n"


def test_frontmatter_keeps_a_value_containing_the_delimiter():
    # `---` inside a value is not a closing delimiter; splitting on it would
    # drop `status` and behead the page at the second occurrence.
    text = "---\ntitle: A --- B\nstatus: verified\n---\n# Title\n"
    frontmatter, body = _split_frontmatter(text)
    assert frontmatter == {"title": "A --- B", "status": "verified"}
    assert body == "# Title\n"


def test_body_opening_with_a_horizontal_rule_is_kept_whole():
    # No frontmatter: the block between the rules is prose, not a YAML mapping,
    # so the page must come back untouched rather than truncated.
    text = "---\n\nSome prose.\n\n---\n\n# Title\n"
    frontmatter, body = _split_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_unterminated_frontmatter_is_kept_whole():
    text = "---\nstatus: verified\n# Title\n"
    frontmatter, body = _split_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_malformed_yaml_keeps_the_page_whole():
    text = "---\nstatus: [unclosed\n---\n# Title\n"
    frontmatter, body = _split_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_page_without_frontmatter_is_unchanged():
    text = "# Title\n\nProse.\n"
    assert _split_frontmatter(text) == ({}, text)
