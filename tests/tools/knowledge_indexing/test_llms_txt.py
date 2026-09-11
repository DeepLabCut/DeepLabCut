from __future__ import annotations

import re
from pathlib import Path

from tools.knowledge_indexing.__main__ import API_BASE_URL, DOCS_VERSION_LABEL, LLMS_TXT, PACKAGE, main
from tools.knowledge_indexing.llms_txt import build_llms_txt
from tools.knowledge_indexing.schemas import LATEST_RELEASE_ALIAS


def test_llms_txt_lists_stable_and_rolling_api_links():
    # The urls are landing pages, carrying the package segment: api-autonav
    # publishes no index at `reference/`, so a link to it 404s.
    text = build_llms_txt(
        docs_base_url="https://example.test/",
        stable_api_reference_url="https://example.test/dev/latest-release/reference/deeplabcut/",
        rolling_api_reference_url="https://example.test/dev/main/reference/deeplabcut/",
        knowledge_base_url="https://example.test/knowledge/",
        version_label="main",
    )

    assert "- [API reference (stable release)](https://example.test/dev/latest-release/reference/deeplabcut/)" in text
    assert "- [API reference (main / unreleased)](https://example.test/dev/main/reference/deeplabcut/)" in text


def test_llms_txt_keeps_machine_readable_links_version_scoped():
    text = build_llms_txt(
        docs_base_url="https://example.test/",
        stable_api_reference_url="https://example.test/dev/latest-release/reference/deeplabcut/",
        rolling_api_reference_url="https://example.test/dev/main/reference/deeplabcut/",
        knowledge_base_url="https://example.test/knowledge/",
        version_label="3.0",
    )

    assert "https://example.test/knowledge/3.0/docs.jsonl" in text
    assert "https://example.test/knowledge/3.0/api.jsonl" in text


def test_cli_points_each_api_link_at_the_right_deploy(tmp_path: Path):
    # The unit tests above are handed both urls, so they cannot catch the two
    # being swapped or built from the wrong label in `__main__.py`.
    (tmp_path / "_toc.yml").write_text("format: jb-book\nroot: README\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# DeepLabCut\n\nLead.\n", encoding="utf-8")
    package = tmp_path / PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text('"""Package."""\n', encoding="utf-8")

    output = tmp_path / "out"
    assert main(["--output", str(output), "--repo", str(tmp_path), "--revision", "abc123"]) == 0

    link = re.compile(r"^- \[(?P<label>[^\]]+)\]\((?P<url>[^)]+)\)")
    lines = {
        m["label"]: m["url"]
        for m in (link.match(line) for line in (output / LLMS_TXT).read_text(encoding="utf-8").splitlines())
        if m and m["label"].startswith("API reference")
    }
    reference = f"reference/{PACKAGE}/"
    assert lines["API reference (stable release)"] == (
        f"{API_BASE_URL.format(version=LATEST_RELEASE_ALIAS)}{reference}"
    )
    assert lines["API reference (main / unreleased)"] == (
        f"{API_BASE_URL.format(version=DOCS_VERSION_LABEL)}{reference}"
    )
