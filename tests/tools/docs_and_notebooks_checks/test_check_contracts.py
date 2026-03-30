from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest


# -----------------------------
# Module loader (tools/ is not necessarily a package)
# -----------------------------
def load_tool_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "docs_and_notebooks_check.py"
    assert tool_path.exists(), f"Missing tool: {tool_path}"

    spec = importlib.util.spec_from_file_location("docs_and_notebooks_check", tool_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="session")
def tool() -> ModuleType:
    return load_tool_module()


# -----------------------------
# Git helpers for a temp repo
# -----------------------------
def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=True)


def _git_init(repo: Path) -> None:
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "ci@example.com"], repo)
    _run(["git", "config", "user.name", "CI"], repo)


def _git_commit(repo: Path, message: str, when_iso: str) -> None:
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = when_iso
    env["GIT_COMMITTER_DATE"] = when_iso
    _run(["git", "add", "-A"], repo, env=env)
    _run(["git", "commit", "-m", message], repo, env=env)


def _write(repo: Path, rel: str, content: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# -----------------------------
# Contract tests
# -----------------------------
def test_marker_constants_exist(tool):
    assert hasattr(tool, "META_COMMIT_MARKER")
    assert hasattr(tool, "SUGGESTED_TAGGED_COMMIT")
    assert tool.META_COMMIT_MARKER in tool.SUGGESTED_TAGGED_COMMIT


def test_schema_contract_fields(tool):
    # DLCMeta must have new fields and must NOT have old last_git_updated
    meta = tool.DLCMeta()
    assert hasattr(meta, "last_content_updated")
    assert hasattr(meta, "last_metadata_updated")
    assert hasattr(meta, "last_verified")
    assert hasattr(meta, "verified_for")
    assert not hasattr(meta, "last_git_updated")


def test_git_content_date_skips_meta_commits(tool, tmp_path: Path):
    """
    Contract: last_content_updated is computed from git history excluding metadata commits.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    # meta-only rewrite (simulated) committed with marker
    _write(
        repo,
        rel,
        "---\ndeeplabcut:\n  last_metadata_updated: 2026-03-01\n---\n# hello\n",
    )
    _git_commit(
        repo,
        f"chore(meta): update {tool.META_COMMIT_MARKER}",
        "2026-03-01T12:00:00+00:00",
    )

    # raw touched date = 2026-03-01
    touched = tool.git_last_touched(repo, rel)
    assert touched == date(2026, 3, 1)

    # content updated date should skip marker commit => 2020-01-01
    content_date, used_fallback = tool.git_last_content_updated(repo, rel)
    assert content_date == date(2020, 1, 1)
    assert used_fallback is False


def test_git_content_date_fallback_when_only_meta_commits(tool, tmp_path: Path):
    """
    If all commits touching the file are meta-marker commits, we fall back to git_last_touched
    and flag used_fallback=True.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    _write(repo, rel, "---\ndeeplabcut:\n  notes: hi\n---\n")
    _git_commit(
        repo,
        f"chore(meta): init {tool.META_COMMIT_MARKER}",
        "2026-03-01T12:00:00+00:00",
    )

    content_date, used_fallback = tool.git_last_content_updated(repo, rel)
    assert content_date == date(2026, 3, 1)
    assert used_fallback is True


def test_scan_is_read_only(tool, tmp_path: Path, monkeypatch):
    """
    Contract: report/check (scan_files) must be read-only.
    We validate by asserting file content does not change.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    orig = "---\ndeeplabcut:\n  last_verified: 2020-01-01\n---\n# hello\n"
    _write(repo, rel, orig)
    _git_commit(repo, "docs: add page", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    before = (repo / rel).read_text(encoding="utf-8")
    records = tool.scan_files(repo, cfg, targets=[rel])
    after = (repo / rel).read_text(encoding="utf-8")

    assert before == after
    assert len(records) == 1
    assert records[0].path == rel
    assert records[0].kind == "md"


def test_update_requires_ack_when_write(tool, tmp_path: Path):
    """
    Contract: write mode should refuse unless --ack-meta-commit-marker is provided.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    # should refuse to write without ack
    with pytest.raises(SystemExit):
        tool.update_files(
            repo_root=repo,
            cfg=cfg,
            targets=[rel],
            write=True,
            set_content_date_from_git=True,
            set_last_verified=None,
            set_verified_for=None,
            ack_meta_commit_marker=False,
        )


def test_update_set_content_date_from_git_only_changes_that_field(tool, tmp_path: Path):
    """
    Contract: update --set-content-date-from-git only sets last_content_updated
    (plus last_metadata_updated when writing),
    does NOT override last_verified/verified_for unless explicitly provided.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    initial = "---\ndeeplabcut:\n  last_verified: 2020-02-02\n  verified_for: 3.0.0rc1\n---\n# hello\n"
    _write(repo, rel, initial)
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    records = tool.update_files(
        repo_root=repo,
        cfg=cfg,
        targets=[rel],
        write=True,
        set_content_date_from_git=True,
        set_last_verified=None,
        set_verified_for=None,
        ack_meta_commit_marker=True,
    )
    assert len(records) == 1

    # Read back and confirm verified fields unchanged
    text = (repo / rel).read_text(encoding="utf-8")
    fm, body, _ = tool.read_md_frontmatter(text)
    assert isinstance(fm, dict) and tool.DLC_NAMESPACE in fm
    meta = fm[tool.DLC_NAMESPACE]

    assert meta["last_verified"] == "2020-02-02"
    assert meta["verified_for"] == "3.0.0rc1"

    # last_content_updated should reflect git content date (2020-01-01)
    assert meta["last_content_updated"] == "2020-01-01"

    # last_metadata_updated should exist because we wrote
    assert "last_metadata_updated" in meta


def test_update_set_verified_fields_only_changes_verified(tool, tmp_path: Path):
    """
    Contract: update with --set-last-verified / --set-verified-for changes only those fields
    (plus last_metadata_updated if writing), and does not set last_content_updated unless requested.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    initial = "---\ndeeplabcut:\n  last_content_updated: 2000-01-01\n---\n# hello\n"
    _write(repo, rel, initial)
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    records = tool.update_files(
        repo_root=repo,
        cfg=cfg,
        targets=[rel],
        write=True,
        set_content_date_from_git=False,
        set_last_verified=date(2026, 3, 5),
        set_verified_for="3.0.0rc13",
        ack_meta_commit_marker=True,
    )
    assert len(records) == 1

    text = (repo / rel).read_text(encoding="utf-8")
    fm, _body, _ = tool.read_md_frontmatter(text)
    meta = fm[tool.DLC_NAMESPACE]

    # Verified fields updated
    assert meta["last_verified"] == "2026-03-05"
    assert meta["verified_for"] == "3.0.0rc13"

    # last_content_updated remains whatever it was (not overwritten)
    assert meta["last_content_updated"] == "2000-01-01"


def test_normalize_is_explicit_and_marks_would_change(tool, tmp_path: Path):
    """
    Contract: normalize is separate and explicit; in dry-run it should mark would_change
    if notebook is not already in canonical nbformat output.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/nbs/nb.ipynb"
    # Minimal notebook JSON but not in nbformat canonical formatting (indent/newline differences)
    raw = '{\n  "cells": [],\n  "metadata": {},\n  "nbformat": 4,\n  "nbformat_minor": 5\n}\n'
    _write(repo, rel, raw)
    _git_commit(repo, "docs: add notebook", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    # Dry-run normalize: should set would_change True if not normalized
    records = tool.normalize_notebooks(
        repo_root=repo,
        cfg=cfg,
        targets=[rel],
        write=False,
        ack_meta_commit_marker=True,
    )
    assert len(records) == 1
    assert records[0].kind == "ipynb"
    # may be True depending on canonical formatting differences
    assert records[0].would_change


def test_write_outputs_contract(tool, tmp_path: Path):
    """
    Contract: write_outputs creates both JSON and Markdown files and JSON is schema-valid.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )
    records = tool.scan_files(repo, cfg, targets=[rel])

    report = tool.Report(
        generated_at=datetime.now(timezone.utc),
        repo_root=str(repo),
        config_path="in-memory",
        totals=tool.summarize(records),
        records=records,
    )

    out_dir = tmp_path / "out"
    json_path, md_path = tool.write_outputs(report, cfg, out_dir)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert "records" in payload and isinstance(payload["records"], list)
    assert md_path.read_text(encoding="utf-8").startswith("#")


def test_notebook_missing_dlc_namespace_warns_missing_metadata(tool, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/nbs/nb.ipynb"
    # Valid minimal notebook, but no "deeplabcut" namespace under metadata
    nb = '{\n  "cells": [],\n  "metadata": {},\n  "nbformat": 4,\n  "nbformat_minor": 5\n}\n'
    _write(repo, rel, nb)
    _git_commit(repo, "docs: add notebook", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    records = tool.scan_files(repo, cfg, targets=[rel])
    assert len(records) == 1
    r = records[0]
    assert r.kind == "ipynb"
    assert "missing_metadata" in r.warnings
    assert r.meta is None


def test_notebook_invalid_dlc_namespace_warns_invalid_metadata(tool, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)

    rel = "docs/nbs/nb.ipynb"
    # deeplabcut namespace exists but is invalid: last_verified must be a date
    nb = (
        "{\n"
        '  "cells": [],\n'
        '  "metadata": {\n'
        '    "deeplabcut": {\n'
        '      "last_verified": "not-a-date"\n'
        "    }\n"
        "  },\n"
        '  "nbformat": 4,\n'
        '  "nbformat_minor": 5\n'
        "}\n"
    )
    _write(repo, rel, nb)
    _git_commit(repo, "docs: add notebook with bad meta", "2020-01-01T12:00:00+00:00")

    cfg = tool.ToolConfig(
        version=1,
        scan=tool.ScanConfig(include=[rel], exclude=[]),
        policy=tool.PolicyConfig(),
    )

    records = tool.scan_files(repo, cfg, targets=[rel])
    assert len(records) == 1
    r = records[0]
    assert r.kind == "ipynb"
    assert "invalid_metadata" in r.warnings
    assert r.meta is None
