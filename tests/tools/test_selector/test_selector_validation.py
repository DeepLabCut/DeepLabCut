from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


# -----------------
#  Git helpers
# -----------------
def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    return repo


def _commit_file(repo: Path, relpath: str, content: str, message: str) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relpath)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_event(tmp_path: Path, payload: dict) -> Path:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload), encoding="utf-8")
    return event_path


# --------------
# SHA validation & diff range parsing
# --------------
def test_validate_sha_accepts(selector):
    assert selector._validate_sha("x", "abc1234") == "abc1234"
    assert selector._validate_sha("x", "a" * 40) == "a" * 40


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "notasha",  # non-hex
        "123",  # too short
        "g" * 40,  # non-hex
        " " * 8,  # whitespace
    ],
)
def test_validate_sha_rejects(selector, bad):
    with pytest.raises(ValueError):
        selector._validate_sha("x", bad)


def test_determine_diff_range_pr_uses_merge_base(selector, tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    merge_base = _commit_file(repo, "shared.txt", "base", "base commit")
    base_sha = _commit_file(repo, "main.txt", "main", "main branch commit")

    _git(repo, "checkout", "-b", "feature", merge_base)
    head_sha = _commit_file(repo, "feature.txt", "feature", "feature branch commit")

    event_path = _write_event(
        tmp_path,
        {
            "pull_request": {
                "base": {"sha": base_sha},
                "head": {"sha": head_sha},
            }
        },
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    base, head, mode = selector.determine_diff_range(repo, None, None)

    assert base == merge_base
    assert head == head_sha
    assert mode == selector.DiffMode.PR


def test_determine_diff_range_push_uses_before_after(selector, tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    before = _commit_file(repo, "a.txt", "one", "first commit")
    after = _commit_file(repo, "a.txt", "two", "second commit")

    event_path = _write_event(tmp_path, {"before": before, "after": after})
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    base, head, mode = selector.determine_diff_range(repo, None, None)

    assert base == before
    assert head == after
    assert mode == selector.DiffMode.PUSH


def test_determine_diff_range_push_zero_sha_uses_empty_tree(selector, tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    after = _commit_file(repo, "initial.txt", "hello", "initial commit")
    zero_sha = "0" * 40
    event_path = _write_event(tmp_path, {"before": zero_sha, "after": after})
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    base, head, mode = selector.determine_diff_range(repo, None, None)

    assert base == selector._empty_tree(repo)
    assert head == after
    assert mode == selector.DiffMode.INITIAL


def test_determine_diff_range_fallback_uses_head_parent(selector, tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    prev = _commit_file(repo, "a.txt", "one", "first commit")
    head_sha = _commit_file(repo, "a.txt", "two", "second commit")

    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)

    base, head, mode = selector.determine_diff_range(repo, None, None)

    assert base == prev
    assert head == head_sha
    assert mode == selector.DiffMode.FALLBACK


# -----------------
# Paths
# -----------------
def test_normalize_relpath_basic(selector):
    assert selector._normalize_relpath("docs/index.md") == "docs/index.md"
    assert selector._normalize_relpath("docs\\index.md") == "docs/index.md"


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "   ",  # whitespace
        "/etc/passwd",  # absolute unix
        "C:/Windows/x",  # absolute windows
        "../secret.txt",  # traversal
        "docs/../../x",  # traversal inside
        "a\x00b",  # NUL
    ],
)
def test_normalize_relpath_rejects_bad(selector, bad):
    with pytest.raises(ValueError):
        selector._normalize_relpath(bad)
