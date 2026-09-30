"""Tests for the `toc` subcommand of tools/docs_and_notebooks_check.py.

All repositories and pages here are synthetic fixtures built in `tmp_path`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "docs_and_notebooks_check.py"


@pytest.fixture(autouse=True)
def no_github_env(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def _page(visibility: str | None = None) -> str:
    """Synthetic page, with a `deeplabcut.visibility` frontmatter when given."""
    body = "# Synthetic page\n"
    if visibility is None:
        return body
    return f"---\ndeeplabcut:\n  visibility: {visibility}\n---\n\n{body}"


def _make_repo(tmp_path: Path, toc_files: list[str], pages: dict[str, str]) -> Path:
    """Build a synthetic repo with a `_toc.yml` listing `toc_files` and the given pages."""
    (tmp_path / ".git").mkdir()
    toc = "format: jb-book\nroot: README\nparts:\n  - caption: Synthetic\n    chapters:\n"
    toc += "".join(f"      - file: {f}\n" for f in toc_files)
    (tmp_path / "_toc.yml").write_text(toc, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Synthetic root\n", encoding="utf-8")
    for rel, text in pages.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cfg = tmp_path / "config.yml"
    cfg.write_text(
        "version: 1\nscan:\n  include: ['docs/**/*.md']\n  exclude: ['**/_build/**']\npolicy: {}\n",
        encoding="utf-8",
    )
    return tmp_path


def _issues(tool: ModuleType, repo: Path) -> dict[str, str]:
    cfg = tool.load_config(repo / "config.yml")
    return {i.path: i.severity for i in tool.check_toc(repo, cfg)}


def test_listed_page_passes(tool, tmp_path):
    repo = _make_repo(tmp_path, ["docs/page"], {"docs/page.md": _page()})
    assert _issues(tool, repo) == {}


def test_toc_entry_with_suffix_matches(tool, tmp_path):
    repo = _make_repo(tmp_path, ["docs/page.md"], {"docs/page.md": _page()})
    assert _issues(tool, repo) == {}


@pytest.mark.parametrize("visibility", [None, "online", "orphan"])
def test_unlisted_page_without_off_toc_visibility_fails(tool, tmp_path, visibility):
    repo = _make_repo(tmp_path, [], {"docs/page.md": _page(visibility)})
    assert _issues(tool, repo) == {"docs/page.md": "error"}


@pytest.mark.parametrize("visibility", ["unlisted", "archived", "orphaned"])
def test_unlisted_page_with_off_toc_visibility_passes(tool, tmp_path, visibility):
    repo = _make_repo(tmp_path, [], {"docs/page.md": _page(visibility)})
    assert _issues(tool, repo) == {}


def test_non_string_visibility_is_an_error_not_a_crash(tool, tmp_path):
    repo = _make_repo(tmp_path, [], {"docs/page.md": _page("[orphaned]")})
    [issue] = tool.check_toc(repo, tool.load_config(repo / "config.yml"))
    assert issue.severity == "error"
    assert "invalid visibility" in issue.reason


def test_frontmatter_parse_error_is_a_single_line_error(tool, tmp_path):
    repo = _make_repo(tmp_path, [], {"docs/page.md": "---\ndeeplabcut:\n  visibility: 'x\n---\n"})
    [issue] = tool.check_toc(repo, tool.load_config(repo / "config.yml"))
    assert issue.severity == "error"
    assert "frontmatter_parse_error" in issue.reason
    assert "\n" not in issue.reason


def test_unresolved_repo_root(tool, tmp_path):
    repo = _make_repo(tmp_path, [], {"docs/page.md": _page()})
    unresolved = repo / "docs" / ".."
    assert _issues(tool, unresolved) == {"docs/page.md": "error"}


def test_listed_page_marked_orphaned_warns(tool, tmp_path):
    repo = _make_repo(tmp_path, ["docs/page"], {"docs/page.md": _page("orphaned")})
    assert _issues(tool, repo) == {"docs/page.md": "warning"}


def test_excluded_paths_are_skipped(tool, tmp_path):
    repo = _make_repo(tmp_path, [], {"docs/_build/page.md": _page()})
    assert _issues(tool, repo) == {}


def test_napari_page_missing_from_toc_fails(tool, tmp_path, monkeypatch, capsys):
    """Regression for #3530: a new napari page was merged without a TOC entry."""
    repo = _make_repo(
        tmp_path,
        ["docs/gui/napari_GUI", "docs/gui/napari/basic_usage"],
        {
            "docs/gui/napari_GUI.md": _page(),
            "docs/gui/napari/basic_usage.md": _page(),
            "docs/gui/napari/troubleshooting.md": _page(),
        },
    )
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    summary = repo / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert tool.main(["--config", str(repo / "config.yml"), "toc"]) == 1

    out = capsys.readouterr().out
    assert "::error file=docs/gui/napari/troubleshooting.md,line=1::" in out
    assert "basic_usage" not in out
    assert "docs/gui/napari/troubleshooting.md" in summary.read_text(encoding="utf-8")


def test_warnings_alone_exit_zero(tool, tmp_path, monkeypatch):
    repo = _make_repo(tmp_path, ["docs/page"], {"docs/page.md": _page("orphaned")})
    monkeypatch.chdir(repo)
    assert tool.main(["--config", str(repo / "config.yml"), "toc"]) == 0


def test_runs_as_script(tmp_path):
    """Invoked by file path, as in CI, where `tools` is not importable as a package."""
    repo = _make_repo(tmp_path, [], {"docs/page.md": _page()})
    proc = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--config", str(repo / "config.yml"), "--no-step-summary", "toc"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    assert "docs/page.md" in proc.stdout
