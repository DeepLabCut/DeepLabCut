from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.knowledge_indexing.schemas import API_FILE, DOCS_FILE, KNOWLEDGE_DIR, TOP_MANIFEST, VERSION_MANIFEST
from tools.knowledge_indexing.write import (
    _check_unique_ids,
    _read_json,
    delete_version,
    write_top_manifest,
    write_version,
)


def _sample_api():
    return [
        SimpleNamespace(
            id="api:deeplabcut.demo",
            module="deeplabcut.demo",
            summary="Demo module.",
            source="deeplabcut/demo.py:1",
            docs_url="https://example.test/dev/main/reference/deeplabcut/demo/",
            symbols=(
                SimpleNamespace(
                    name="run",
                    kind="function",
                    summary="Run demo.",
                    signature="run(x: int) -> None",
                    source="deeplabcut/demo.py:10",
                    docs_url="https://example.test/dev/main/reference/deeplabcut/demo/#run",
                ),
            ),
        )
    ]


def _sample_docs():
    return [
        SimpleNamespace(
            id="docs:install",
            title="Install",
            docs_url="https://example.test/docs/install.html",
            source_file="docs/install.md",
            part="Getting Started",
            parent="",
            children=(),
            summary="How to install.",
            status="verified",
            last_verified="2026-01-01",
            related_pages=(),
            labels=(),
            sections=(
                SimpleNamespace(
                    id="docs:install#requirements",
                    title="Requirements",
                    level=2,
                    anchor="requirements",
                    docs_url="https://example.test/docs/install.html#requirements",
                    excerpt="You need Python.",
                ),
            ),
        )
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_check_unique_ids_raises_on_duplicate():
    with pytest.raises(ValueError, match="Duplicate record id"):
        _check_unique_ids("api.jsonl", ["a", "b", "a"])


def test_read_json_missing_returns_none(tmp_path: Path):
    assert _read_json(tmp_path / "missing.json") is None


def test_read_json_malformed_raises(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _read_json(path)


def test_write_version_writes_jsonl_and_manifest(tmp_path: Path):
    knowledge_dir = tmp_path / KNOWLEDGE_DIR
    api_count, docs_count = write_version(
        knowledge_dir,
        "main",
        _sample_api(),
        _sample_docs(),
        package_version="3.1.0rc1",
        revision="abc123",
    )

    assert api_count == 2
    assert docs_count == 2

    version_dir = knowledge_dir / "main"
    api_rows = _read_jsonl(version_dir / API_FILE)
    docs_rows = _read_jsonl(version_dir / DOCS_FILE)
    manifest = json.loads((version_dir / VERSION_MANIFEST).read_text(encoding="utf-8"))

    assert {row["id"] for row in api_rows} == {"api:deeplabcut.demo", "api:deeplabcut.demo.run"}
    assert all(len(row["content_hash"]) == 64 for row in api_rows)
    assert {row["id"] for row in docs_rows} == {"docs:install", "docs:install#requirements"}
    assert manifest["api_version_label"] == "main"
    assert manifest["api"]["package_version"] == "3.1.0rc1"
    assert manifest["api"]["revision"] == "abc123"
    assert manifest["docs"]["revision"] == "abc123"


def test_write_version_skip_keeps_existing_half(tmp_path: Path):
    knowledge_dir = tmp_path / KNOWLEDGE_DIR
    write_version(
        knowledge_dir,
        "main",
        _sample_api(),
        _sample_docs(),
        package_version="1.0",
        revision="first",
    )
    first = json.loads((knowledge_dir / "main" / VERSION_MANIFEST).read_text(encoding="utf-8"))

    write_version(
        knowledge_dir,
        "main",
        None,
        _sample_docs(),
        package_version="2.0",
        revision="second",
    )
    second = json.loads((knowledge_dir / "main" / VERSION_MANIFEST).read_text(encoding="utf-8"))

    assert second["api"] == first["api"]
    assert second["docs"]["revision"] == "second"


def test_write_top_manifest_and_delete_version(tmp_path: Path):
    knowledge_dir = tmp_path / KNOWLEDGE_DIR
    write_version(knowledge_dir, "main", _sample_api(), _sample_docs(), revision="r1")
    write_version(knowledge_dir, "3.0", _sample_api(), None, revision="r2")
    write_top_manifest(knowledge_dir, docs_version_label="main")

    top = json.loads((knowledge_dir / TOP_MANIFEST).read_text(encoding="utf-8"))
    assert top["docs"]["path"] == f"main/{DOCS_FILE}"
    assert top["api"]["versions"] == ["3.0", "main"]

    delete_version(knowledge_dir, "3.0")
    write_top_manifest(knowledge_dir, docs_version_label="main")

    assert not (knowledge_dir / "3.0").exists()
    assert (knowledge_dir / "main" / DOCS_FILE).is_file()
    top_after = json.loads((knowledge_dir / TOP_MANIFEST).read_text(encoding="utf-8"))
    assert top_after["api"]["versions"] == ["main"]


def test_top_manifest_skips_version_dir_without_api_file(tmp_path: Path):
    knowledge_dir = tmp_path / KNOWLEDGE_DIR
    write_version(knowledge_dir, "main", _sample_api(), None, revision="r1")
    (knowledge_dir / "ghost").mkdir()
    write_top_manifest(knowledge_dir, docs_version_label="main")

    top = json.loads((knowledge_dir / TOP_MANIFEST).read_text(encoding="utf-8"))
    assert top["api"]["versions"] == ["main"]


def test_delete_main_via_cli_refused(tmp_path: Path, capsys):
    from tools.knowledge_indexing.__main__ import main

    knowledge_dir = tmp_path / KNOWLEDGE_DIR
    write_version(knowledge_dir, "main", _sample_api(), None, revision="r1")

    code = main(["--delete", "--version-label", "main", "--output", str(tmp_path)])
    assert code == 1
    assert "cannot be deleted" in capsys.readouterr().err
    assert (knowledge_dir / "main").is_dir()
