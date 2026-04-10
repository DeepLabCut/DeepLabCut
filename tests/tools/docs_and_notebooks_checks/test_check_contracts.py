from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from collections.abc import Callable
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


def _write_default_cfg(repo: Path, include: list[str]) -> Path:
    cfg_path = repo / "tools" / "docs_and_notebooks_report_config.yml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "version: 1\n"
        "scan:\n"
        "  include:\n" + "".join(f"    - {pat}\n" for pat in include) + "  exclude: []\n"
        "policy:\n"
        "  warn_if_content_older_than_days: 365\n"
        "  warn_if_verified_older_than_days: 365\n"
        "  missing_last_verified_is_warning: true\n"
        "  fail_on_scan_errors: false\n"
        "  require_metadata: []\n"
        "  require_recent_verification: []\n"
        "  require_notebook_normalized: []\n",
        encoding="utf-8",
    )
    return cfg_path


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
# Shared fixtures
# -----------------------------
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_init(repo)
    return repo


@pytest.fixture
def cfg(tool) -> Callable[..., object]:
    def _make_cfg(
        include: list[str],
        exclude: list[str] | None = None,
        **policy_overrides,
    ):
        policy = tool.PolicyConfig(**policy_overrides)
        return tool.ToolConfig(
            version=1,
            scan=tool.ScanConfig(include=include, exclude=exclude or []),
            policy=policy,
        )

    return _make_cfg


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


def test_git_content_date_skips_meta_commits(tool, repo: Path):
    """
    Contract: last_content_updated is computed from git history excluding metadata commits.
    """
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


def test_git_content_date_fallback_when_only_meta_commits(tool, repo: Path):
    """
    If all commits touching the file are meta-marker commits, we fall back to git_last_touched
    and flag used_fallback=True.
    """
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


def test_scan_is_read_only(tool, repo: Path, cfg):
    """
    Contract: report/check (scan_files) must be read-only.
    We validate by asserting file content does not change.
    """
    rel = "docs/page.md"
    orig = "---\ndeeplabcut:\n  last_verified: 2020-01-01\n---\n# hello\n"
    _write(repo, rel, orig)
    _git_commit(repo, "docs: add page", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    before = (repo / rel).read_text(encoding="utf-8")
    records = tool.scan_files(repo, tool_cfg, targets=[r".\docs\page.md"])
    after = (repo / rel).read_text(encoding="utf-8")

    assert before == after
    assert len(records) == 1
    assert records[0].path == rel
    assert records[0].kind == "md"


def test_scan_targets_support_directory_and_glob(tool, repo: Path, cfg):
    rel_a = "docs/gui/napari/basic_usage.md"
    rel_b = "docs/gui/napari/advanced_usage.md"
    rel_c = "docs/other/overview.md"

    _write(repo, rel_a, "# a\n")
    _write(repo, rel_b, "# b\n")
    _write(repo, rel_c, "# c\n")
    _git_commit(repo, "docs: add pages", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=["docs/**/*.md"])

    # Directory selector
    recs_dir = tool.scan_files(repo, tool_cfg, targets=["docs/gui/napari/"])
    paths_dir = sorted(r.path for r in recs_dir)
    assert set(paths_dir) == {rel_a, rel_b}

    # Glob selector
    recs_glob = tool.scan_files(repo, tool_cfg, targets=["docs/gui/napari/*.md"])
    paths_glob = sorted(r.path for r in recs_glob)
    assert set(paths_glob) == {rel_a, rel_b}

    # Recursive glob selector
    recs_recursive = tool.scan_files(repo, tool_cfg, targets=["docs/**/*.md"])
    paths_recursive = sorted(r.path for r in recs_recursive)
    assert set(paths_recursive) == {rel_a, rel_b, rel_c}


def test_validate_requested_targets_treats_dot_slash_as_unmatched(tool, repo: Path, cfg):
    _write(repo, "docs/page.md", "# hello\n")
    tool_cfg = cfg(include=["docs/**/*.md"])

    matched, unmatched = tool.validate_requested_targets(repo, tool_cfg, ["./"])
    assert matched == []
    assert unmatched == ["./"]


def test_validate_requested_targets_treats_empty_like_unmatched(tool, repo: Path, cfg):
    _write(repo, "docs/page.md", "# hello\n")
    tool_cfg = cfg(include=["docs/**/*.md"])

    matched, unmatched = tool.validate_requested_targets(repo, tool_cfg, ["", "   "])
    assert matched == []
    assert unmatched == ["", "   "]


def test_validate_requested_targets_reports_mixed_valid_and_invalid_targets(tool, repo: Path, cfg):
    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    tool_cfg = cfg(include=["docs/**/*.md"])

    matched, unmatched = tool.validate_requested_targets(repo, tool_cfg, [rel, "./"])
    assert rel in matched
    assert unmatched == ["./"]


def test_scan_files_with_invalid_only_targets_matches_nothing(tool, repo: Path, cfg):
    tool_cfg = cfg(include=["docs/**/*.md"])
    records = tool.scan_files(repo, tool_cfg, targets=["./"])
    assert records == []


def test_validate_requested_targets_reports_unmatched(tool, repo: Path, cfg):
    rel_a = "docs/gui/napari/basic_usage.md"
    rel_b = "docs/gui/napari/advanced_usage.md"

    _write(repo, rel_a, "# a\n")
    _write(repo, rel_b, "# b\n")
    _git_commit(repo, "docs: add pages", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=["docs/**/*.md"])

    matched, unmatched = tool.validate_requested_targets(
        repo,
        tool_cfg,
        targets=[
            r".\docs\gui\napari\basic_usage.md",
            "docs/gui/napari/",
            "docs/**/*.md",
            "docs/missing/",
            "examples/**/*.ipynb",
        ],
    )

    assert matched == sorted([rel_a, rel_b])
    assert unmatched == ["docs/missing/", "examples/**/*.ipynb"]


def test_update_requires_ack_when_write(tool, repo: Path, cfg):
    """
    Contract: write mode should refuse unless --ack-meta-commit-marker is provided.
    """
    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    # should refuse to write without ack
    with pytest.raises(SystemExit):
        tool.update_files(
            repo_root=repo,
            cfg=tool_cfg,
            targets=[rel],
            write=True,
            set_content_date_from_git=True,
            set_last_verified=None,
            set_verified_for=None,
            ack_meta_commit_marker=False,
        )


def test_update_set_content_date_from_git_only_changes_that_field(tool, repo: Path, cfg):
    """
    Contract: update --set-content-date-from-git only sets last_content_updated
    (plus last_metadata_updated when writing),
    does NOT override last_verified/verified_for unless explicitly provided.
    """
    rel = "docs/page.md"
    initial = "---\ndeeplabcut:\n  last_verified: 2020-02-02\n  verified_for: 3.0.0rc1\n---\n# hello\n"
    _write(repo, rel, initial)
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    records = tool.update_files(
        repo_root=repo,
        cfg=tool_cfg,
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
    fm, _body, _ = tool.read_md_frontmatter(text)
    assert isinstance(fm, dict) and tool.DLC_NAMESPACE in fm
    meta = fm[tool.DLC_NAMESPACE]

    assert meta["last_verified"] == "2020-02-02"
    assert meta["verified_for"] == "3.0.0rc1"

    # last_content_updated should reflect git content date (2020-01-01)
    assert meta["last_content_updated"] == "2020-01-01"

    # last_metadata_updated should exist because we wrote
    assert "last_metadata_updated" in meta


def test_update_set_verified_fields_only_changes_verified(tool, repo: Path, cfg):
    """
    Contract: update with --set-last-verified / --set-verified-for changes only those fields
    (plus last_metadata_updated if writing), and does not set last_content_updated unless requested.
    """
    rel = "docs/page.md"
    initial = "---\ndeeplabcut:\n  last_content_updated: 2000-01-01\n---\n# hello\n"
    _write(repo, rel, initial)
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    records = tool.update_files(
        repo_root=repo,
        cfg=tool_cfg,
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


def test_normalize_is_explicit_and_marks_would_change(tool, repo: Path, cfg):
    """
    Contract: normalize is separate and explicit; in dry-run it should mark would_change
    if notebook is not already in canonical nbformat output.
    """
    rel = "docs/nbs/nb.ipynb"
    # Minimal notebook JSON but not in nbformat canonical formatting (indent/newline differences)
    raw = '{\n  "cells": [],\n  "metadata": {},\n  "nbformat": 4,\n  "nbformat_minor": 5\n}\n'
    _write(repo, rel, raw)
    _git_commit(repo, "docs: add notebook", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    # Dry-run normalize: should set would_change True if not normalized
    records = tool.normalize_notebooks(
        repo_root=repo,
        cfg=tool_cfg,
        targets=[rel],
        write=False,
        ack_meta_commit_marker=True,
    )
    assert len(records) == 1
    assert records[0].kind == "ipynb"
    # may be True depending on canonical formatting differences
    assert records[0].would_change


def test_write_outputs_contract(tool, repo: Path, cfg, tmp_path: Path):
    """
    Contract: write_outputs creates both JSON and Markdown files and JSON is schema-valid.
    """
    rel = "docs/page.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: initial content", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])
    records = tool.scan_files(repo, tool_cfg, targets=[rel])

    report = tool.Report(
        generated_at=datetime.now(timezone.utc),
        repo_root=str(repo),
        config_path="in-memory",
        totals=tool.summarize(records),
        records=records,
    )

    out_dir = tmp_path / "out"
    json_path, md_path = tool.write_outputs(report, tool_cfg, out_dir)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert "records" in payload and isinstance(payload["records"], list)
    assert md_path.read_text(encoding="utf-8").startswith("#")


def test_notebook_missing_dlc_namespace_warns_missing_metadata(tool, repo: Path, cfg):
    rel = "docs/nbs/nb.ipynb"
    # Valid minimal notebook, but no "deeplabcut" namespace under metadata
    nb = '{\n  "cells": [],\n  "metadata": {},\n  "nbformat": 4,\n  "nbformat_minor": 5\n}\n'
    _write(repo, rel, nb)
    _git_commit(repo, "docs: add notebook", "2020-01-01T12:00:00+00:00")

    tool_cfg = cfg(include=[rel])

    records = tool.scan_files(repo, tool_cfg, targets=[rel])
    assert len(records) == 1
    r = records[0]
    assert r.kind == "ipynb"
    assert "missing_metadata" in r.warnings
    assert r.meta is None


def test_notebook_invalid_dlc_namespace_warns_invalid_metadata(tool, repo: Path, cfg):
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

    tool_cfg = cfg(include=[rel])

    records = tool.scan_files(repo, tool_cfg, targets=[rel])
    assert len(records) == 1
    r = records[0]
    assert r.kind == "ipynb"
    assert "invalid_metadata" in r.warnings
    assert r.meta is None


def test_main_prints_matched_files_and_fails_on_unmatched_targets(tool, repo: Path, monkeypatch, capsys):
    rel = "docs/gui/napari/basic_usage.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: add page", "2020-01-01T12:00:00+00:00")

    cfg_path = _write_default_cfg(repo, include=["docs/**/*.md"])

    monkeypatch.chdir(repo)

    rc = tool.main(
        [
            "--config",
            str(cfg_path),
            "report",
            "--targets",
            r".\docs\gui\napari\basic_usage.md",
            "docs/missing/",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 2
    assert "Matched 1 file(s) from --targets:" in out
    assert f"- {rel}" in out
    assert "Unmatched --targets:" in out
    assert "- docs/missing/" in out


def test_main_prints_matched_files_for_valid_targets(tool, repo: Path, monkeypatch, capsys):
    rel = "docs/gui/napari/basic_usage.md"
    _write(repo, rel, "# hello\n")
    _git_commit(repo, "docs: add page", "2020-01-01T12:00:00+00:00")
    cfg_path = _write_default_cfg(repo, include=["docs/**/*.md"])

    monkeypatch.chdir(repo)

    rc = tool.main(
        [
            "--config",
            str(cfg_path),
            "report",
            "--targets",
            "docs/gui/napari/",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Matched 1 file(s) from --targets:" in out
    assert f"- {rel}" in out
    assert "Report generated:" in out


def test_main_returns_2_for_invalid_target_selector(tool, repo: Path, monkeypatch):
    _write(repo, "docs/page.md", "# hello\n")
    _git_commit(repo, "docs: add page", "2020-01-01T12:00:00+00:00")
    cfg_path = _write_default_cfg(repo, include=["docs/**/*.md"])

    monkeypatch.chdir(repo)

    rc = tool.main(["--config", str(cfg_path), "report", "--targets", "./"])
    assert rc == 2
