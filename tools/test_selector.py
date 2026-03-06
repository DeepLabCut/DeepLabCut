#!/usr/bin/env python3
"""Deterministic, strictly validated test selector for DeepLabCut.

Outputs a single, unambiguous *plan* enum plus structured lists:

  - plan: one of {docs_only, fast, full}
  - pytest_paths: JSON list of pytest path arguments
  - functional_scripts: JSON list of python script paths

Safety principles
-----------------
- Fail-safe: if changes cannot be determined or are ambiguous, plan=full.
- Deterministic: derives diff range from GitHub Actions event payload when available.
  * pull_request: uses merge-base(base.sha, head.sha) .. head.sha
  * push: uses before .. after
  * fallback: attempts HEAD~1 .. HEAD
- Secure: never emits shell command strings; only structured data.
- Strict: Pydantic schema validation (extra=forbid), SHA validation, path sanitization.

Intended usage in GitHub Actions
-------------------------------
- Checkout with sufficient history for merge-base/diff (typically fetch-depth: 0).
- Run:
    python tools/test_selector.py --write-github-output --json

This will write the following keys to $GITHUB_OUTPUT:
  plan, pytest_paths, functional_scripts, reasons, changed_files

Notes
-----
- This script intentionally keeps the routing rules simple and location-based.
- Extend CATEGORY_RULES and FULL_SUITE_TRIGGERS as needed, keeping rules auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

DLC_NAMESPACE = "deeplabcut"


class Plan(str, Enum):
    """Unambiguous test plan."""

    DOCS_ONLY = "docs_only"  # Docs build only
    FAST = "fast"  # Targeted pytest + optional functional scripts (single-lane)
    FULL = "full"  # Delegate to full test workflow/matrix


class SelectorResult(BaseModel):
    """Strict output schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    plan: Plan

    pytest_paths: List[str] = Field(default_factory=list)
    functional_scripts: List[str] = Field(default_factory=list)

    # Audit fields
    reasons: List[str] = Field(default_factory=list)
    changed_files: List[str] = Field(default_factory=list)


SelectorResult.model_rebuild()  # Ensure model is fully built at import time for validation in main()

# -----------------------------
# Configuration (simple, auditable)
# -----------------------------
MINIMAL_PYTEST = ["tests/test_auxiliaryfunctions.py"]


# Conservative full-suite triggers: if any changed file matches, plan=FULL.
FULL_SUITE_TRIGGERS: List[Callable[[str], bool]] = [
    lambda p: p.startswith("tests/"),
    lambda p: p == "pyproject.toml",
    lambda p: p.endswith(".lock"),
    # lambda p: p.endswith("requirements.txt"), # outdated
    lambda p: p.endswith("environment.yml"),
]


# Category rules are location-based. If multiple categories match, we treat it as mixed.
# For prototype simplicity:
#   - docs-only if *all* files are docs-like
#   - if >2 categories match -> FULL
#   - else -> FAST with merged selections
# TODO @C-Achard Refine selection and rules
CATEGORY_RULES = [
    {
        "name": "docs",
        "match_any": [
            lambda p: p.startswith("docs/"),
            lambda p: p.endswith(".md"),
            lambda p: p.endswith(".rst"),
            lambda p: p in {"_config.yml", "_toc.yml"},
        ],
        "pytest_paths": [],
        "functional_scripts": [],
    },
    {
        "name": "notebooks_examples",
        "match_any": [
            lambda p: p.startswith("examples/JUPYTER/"),
            lambda p: p.startswith("examples/COLAB/"),
            lambda p: p.endswith(".ipynb"),
        ],
        "pytest_paths": MINIMAL_PYTEST,
        "functional_scripts": [],
    },
    {
        "name": "superanimal_modelzoo",
        "match_any": [
            lambda p: p.startswith("deeplabcut/modelzoo/"),
            lambda p: "superanimal" in p.lower(),
            lambda p: "modelzoo" in p.lower(),
        ],
        "pytest_paths": [
            "tests/test_predict_supermodel.py",
            "tests/pose_estimation_pytorch/modelzoo/",
            "tests/pose_estimation_pytorch/other/test_modelzoo.py",
        ],
        "functional_scripts": [],
    },
    {
        "name": "multianimal",
        "match_any": [
            lambda p: "multianimal" in p.lower(),
            lambda p: p.startswith("deeplabcut/pose_estimation_tensorflow/")
            and "multi" in p.lower(),
            lambda p: p.startswith("deeplabcut/pose_estimation_pytorch/")
            and "multi" in p.lower(),
        ],
        "pytest_paths": [
            "tests/test_auxfun_multianimal.py",
            "tests/test_pose_multianimal_imgaug.py",
            "tests/test_predict_multianimal.py",
        ],
        "functional_scripts": [
            "examples/testscript_multianimal.py",
        ],
    },
    {
        "name": "core",
        "match_any": [
            lambda p: p.startswith("deeplabcut/core/"),
            lambda p: p.startswith("deeplabcut/utils/"),
            lambda p: p.startswith("deeplabcut/pose_estimation_tensorflow/"),
            lambda p: p.startswith("deeplabcut/pose_estimation_pytorch/"),
            lambda p: p == "deeplabcut/auxiliaryfunctions.py",
        ],
        "pytest_paths": [
            "tests/test_auxiliaryfunctions.py",
            "tests/core/",
            "tests/utils/",
            "tests/pose_estimation_pytorch/",
        ],
        "functional_scripts": [],
    },
    {
        "name": "ci_tools",
        "match_any": [
            lambda p: p.startswith(".github/"),
            lambda p: p.startswith("tools/"),
        ],
        "pytest_paths": MINIMAL_PYTEST,
        "functional_scripts": [],
    },
]


# -----------------------------
# Git helpers
# -----------------------------
def _run_git(args: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def find_repo_root() -> Path:
    out = _run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    return Path(out).resolve()


def _validate_sha(label: str, sha: str) -> str:
    if not sha or not SHA_RE.match(sha):
        raise ValueError(f"Invalid {label} SHA: {sha!r}")
    return sha


def _ensure_commit_exists(sha: str, cwd: Path) -> None:
    _run_git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd)


def _load_github_event() -> Dict[str, Any]:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_relpath(p: str) -> str:
    """Normalize and validate a repo-relative path from git output."""
    if "\x00" in p:
        raise ValueError("NUL byte in path")
    p = p.strip().replace("\\", "/")
    if not p:
        raise ValueError("Empty path")
    if p.startswith("/") or re.match(r"^[A-Za-z]:/", p):
        raise ValueError(f"Absolute path not allowed: {p}")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if any(x == ".." for x in parts):
        raise ValueError(f"Path traversal not allowed: {p}")
    return "/".join(parts)


def determine_diff_range(
    repo: Path, override_base: Optional[str], override_head: Optional[str]
) -> Tuple[str, str, str]:
    """Return (base_commit, head_commit, mode)."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = _load_github_event()

    if override_base and override_head:
        base = _validate_sha("base", override_base)
        head = _validate_sha("head", override_head)
        _ensure_commit_exists(base, repo)
        _ensure_commit_exists(head, repo)
        merge_base = _run_git(["merge-base", head, base], repo)
        merge_base = _validate_sha("merge-base", merge_base)
        _ensure_commit_exists(merge_base, repo)
        return merge_base, head, "manual"

    if event_name == "pull_request" and "pull_request" in event:
        base_sha = _validate_sha("base", event["pull_request"]["base"]["sha"])
        head_sha = _validate_sha("head", event["pull_request"]["head"]["sha"])
        _ensure_commit_exists(base_sha, repo)
        _ensure_commit_exists(head_sha, repo)
        # Use merge-base to approximate the PR triple-dot diff base deterministically.
        merge_base = _run_git(["merge-base", head_sha, base_sha], repo)
        merge_base = _validate_sha("merge-base", merge_base)
        _ensure_commit_exists(merge_base, repo)
        return merge_base, head_sha, "pr"

    if event_name == "push" and "before" in event and "after" in event:
        before = _validate_sha("before", event["before"])
        after = _validate_sha("after", event["after"])
        _ensure_commit_exists(before, repo)
        _ensure_commit_exists(after, repo)
        return before, after, "push"

    # Fallback: HEAD~1..HEAD
    try:
        head = _validate_sha("HEAD", _run_git(["rev-parse", "HEAD"], repo))
        prev = _validate_sha("HEAD~1", _run_git(["rev-parse", "HEAD~1"], repo))
        _ensure_commit_exists(prev, repo)
        _ensure_commit_exists(head, repo)
        return prev, head, "fallback"
    except Exception:
        return "", "", "fallback"


def changed_files(repo: Path, base: str, head: str) -> List[str]:
    if not base or not head:
        return []
    out = _run_git(["diff", "--name-only", "--diff-filter=ACMRT", base, head], repo)
    files = [_normalize_relpath(line) for line in out.splitlines() if line.strip()]
    return sorted(set(files))


def _is_safe_relpath(p: str) -> bool:
    """Safety check for a git-relative path: no absolute, no traversal, no NUL."""
    return (
        p
        and "\x00" not in p
        and not p.startswith("/")
        and not re.match(r"^[A-Za-z]:/", p)
        and ".." not in Path(p).parts
    )


def validate_selected_paths(res: SelectorResult, repo: Path) -> SelectorResult:
    missing = []

    # validate pytest paths (files/dirs)
    for p in res.pytest_paths:
        if not _is_safe_relpath(p) or not (repo / p).exists():
            missing.append(f"pytest:{p}")

    # validate functional scripts (files)
    for s in res.functional_scripts:
        if not _is_safe_relpath(s) or not (repo / s).exists():
            missing.append(f"script:{s}")

    if missing:
        # Fail-safe escalation to FULL
        return SelectorResult(
            plan=Plan.FULL,
            pytest_paths=[],
            functional_scripts=[],
            reasons=res.reasons + ["missing_selected_paths"] + missing,
            changed_files=res.changed_files,
        )

    return res


# -----------------------------
# Decision logic
# -----------------------------


def _matches_any(path: str, preds: Sequence[Callable[[str], bool]]) -> bool:
    for pred in preds:
        try:
            if pred(path):
                return True
        except Exception:
            continue
    return False


def decide(files: List[str]) -> SelectorResult:
    reasons: List[str] = []

    if not files:
        # Fail-safe
        return SelectorResult(
            plan=Plan.FULL,
            pytest_paths=[],
            functional_scripts=[],
            reasons=["no_changed_files_or_diff_unavailable"],
            changed_files=[],
        )

    # Full-suite triggers
    if any(_matches_any(f, FULL_SUITE_TRIGGERS) for f in files):
        reasons.append("full_suite_trigger")
        return SelectorResult(
            plan=Plan.FULL,
            pytest_paths=[],
            functional_scripts=[],
            reasons=reasons,
            changed_files=files,
        )

    # docs-only if ALL files match docs category
    docs_rule = next((r for r in CATEGORY_RULES if r["name"] == "docs"), None)
    if docs_rule and all(_matches_any(f, docs_rule["match_any"]) for f in files):
        return SelectorResult(
            plan=Plan.DOCS_ONLY,
            pytest_paths=[],
            functional_scripts=[],
            reasons=["docs_only"],
            changed_files=files,
        )

    # Find matching categories
    matched = []
    for rule in CATEGORY_RULES:
        if any(_matches_any(f, rule["match_any"]) for f in files):
            matched.append(rule)

    if not matched:
        return SelectorResult(
            plan=Plan.FULL,
            pytest_paths=[],
            functional_scripts=[],
            reasons=["no_category_matched"],
            changed_files=files,
        )

    if len(matched) > 2:
        return SelectorResult(
            plan=Plan.FULL,
            pytest_paths=[],
            functional_scripts=[],
            reasons=[f"too_many_categories:{len(matched)}"],
            changed_files=files,
        )

    pytest_paths_set: Set[str] = set()
    functional_set: Set[str] = set()
    for rule in matched:
        reasons.append(f"category:{rule['name']}")
        pytest_paths_set.update(rule.get("pytest_paths", []))
        functional_set.update(rule.get("functional_scripts", []))

    # Ensure at least minimal pytest
    if not pytest_paths_set and not functional_set:
        pytest_paths_set.update(MINIMAL_PYTEST)
        reasons.append("fallback_minimal_pytest")

    return SelectorResult(
        plan=Plan.FAST,
        pytest_paths=sorted(pytest_paths_set),
        functional_scripts=sorted(functional_set),
        reasons=reasons,
        changed_files=files,
    )


# -----------------------------
# Outputs
# -----------------------------
def _render_decision_markdown(res: SelectorResult, limit: int = 40) -> str:
    def bullet(items: List[str], limit_: int = limit) -> str:
        if not items:
            return "_(none)_"
        shown = items[:limit_]
        s = "\n".join(f"- `{x}`" for x in shown)
        if len(items) > limit_:
            s += f"\n- … and {len(items) - limit_} more"
        return s

    badge = {
        Plan.DOCS_ONLY: "📚 **docs_only**",
        Plan.FAST: "⚡ **fast**",
        Plan.FULL: "🧪 **full**",
    }.get(res.plan, f"❓ **{res.plan}**")

    md: List[str] = []
    md.append("# Intelligent test selection\n")
    md.append(f"**Decision (plan):** {badge}\n")
    md.append("## Reasons\n")
    md.append(bullet(res.reasons))
    md.append("\n## Changed files\n")
    md.append(bullet(res.changed_files))
    md.append("\n## Selected pytest paths\n")
    md.append(bullet(res.pytest_paths))
    md.append("\n## Selected functional scripts\n")
    md.append(bullet(res.functional_scripts))
    md.append("")
    return "\n".join(md)


def write_report_files(res: SelectorResult, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "selection.json"
    md_path = out_dir / "decision.md"

    json_path.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_decision_markdown(res), encoding="utf-8")
    return json_path, md_path


def append_job_summary(md_path: Path) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    # Append markdown to the GitHub Actions Job Summary
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(md_path.read_text(encoding="utf-8"))
        f.write("\n")


def write_github_output(res: SelectorResult) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        raise RuntimeError("GITHUB_OUTPUT is not set")

    def j(v) -> str:
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"plan={res.plan.value}\n")
        f.write(f"pytest_paths={j(res.pytest_paths)}\n")
        f.write(f"functional_scripts={j(res.functional_scripts)}\n")
        f.write(f"reasons={j(res.reasons)}\n")
        f.write(f"changed_files={j(res.changed_files)}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic DeepLabCut test selector")
    ap.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    ap.add_argument(
        "--write-github-output",
        action="store_true",
        help="Write outputs to $GITHUB_OUTPUT",
    )
    ap.add_argument("--base-sha", default=None, help="Override base SHA (advanced)")
    ap.add_argument("--head-sha", default=None, help="Override head SHA (advanced)")

    # NEW:
    ap.add_argument(
        "--report-dir",
        default="tmp/test-selection",
        help="Directory to write decision report files (selection.json, decision.md)",
    )
    ap.add_argument(
        "--write-summary",
        action="store_true",
        help="Append decision.md to GitHub Actions Job Summary if available",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    repo = find_repo_root()

    base, head, mode = determine_diff_range(repo, args.base_sha, args.head_sha)
    files = changed_files(repo, base, head)

    res = decide(files)
    res = validate_selected_paths(res, repo)

    # Strict validation
    try:
        res = SelectorResult.model_validate(res.model_dump())
    except ValidationError as e:
        raise RuntimeError(f"Output validation failed: {e}") from e

    # NEW: Always write report files for transparency
    report_dir = Path(args.report_dir)
    json_path, md_path = write_report_files(res, report_dir)

    if args.json:
        print(res.model_dump_json(indent=2))

    if args.write_github_output:
        write_github_output(res)

    # NEW: Write Job Summary (GitHub renders markdown from $GITHUB_STEP_SUMMARY)
    if args.write_summary:
        append_job_summary(md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
