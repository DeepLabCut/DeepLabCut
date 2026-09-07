from __future__ import annotations

from markdown_it import MarkdownIt

from tools.knowledge_indexing.docs_index import _read_structure, _split_frontmatter

PAGE_URL = "https://example.test/docs/page.html"


def _sections(markdown: str):
    tokens = MarkdownIt("commonmark").parse(markdown)
    _, _, sections = _read_structure(tokens, "docs:page", PAGE_URL)
    return sections


def test_only_the_first_h1_is_the_page_title():
    # Some pages use h1 throughout, so only the first may be taken as the title.
    sections = _sections("# Title\n\nLead.\n\n# Second\n\nA.\n\n# Third\n\nB.\n")

    assert [s.title for s in sections] == ["Second", "Third"]
    assert [s.level for s in sections] == [1, 1]


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
