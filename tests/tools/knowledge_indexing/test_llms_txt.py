from __future__ import annotations

from tools.knowledge_indexing.llms_txt import build_llms_txt


def test_llms_txt_lists_stable_and_rolling_api_links():
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
