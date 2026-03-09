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
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)

# DLC_NAMESPACE = "deeplabcut"


class DiffMode(str, Enum):
    """How the diff was determined, for auditing and reporting."""

    PR = "pr"  # merge-base(base, head) .. head
    PUSH = "push"  # before .. after
    MANUAL = "manual"  # CLI override
    FALLBACK = "fallback"  # HEAD^ .. HEAD
    INITIAL = "initial"  # empty tree .. HEAD
    FALLBACK_NO_HEAD = "fallback_no_head"  # couldn't resolve HEAD


MODE_LABELS = {
    DiffMode.PR: "Pull request (merge-base..HEAD)",
    DiffMode.PUSH: "Push (before..after)",
    DiffMode.MANUAL: "Manual override",
    DiffMode.FALLBACK: "Fallback (HEAD^..HEAD)",
    DiffMode.INITIAL: "Initial commit (empty tree..HEAD)",
    DiffMode.FALLBACK_NO_HEAD: "Fallback (couldn't resolve HEAD)",
}


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
    diff_mode: DiffMode = DiffMode.FALLBACK_NO_HEAD

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
# Replace FULL_SUITE_TRIGGERS with a labeled list
FULL_SUITE_TRIGGERS = [
    ("tests/ changed", lambda p: p.startswith("tests/")),
    ("pyproject.toml changed", lambda p: p == "pyproject.toml"),
    ("lockfile changed", lambda p: p.endswith(".lock")),
    ("DEEPLABCUT.yaml changed", lambda p: p.endswith("DEEPLABCUT.yaml")),
]

# Files that should be enforced by dedicated lint workflows, not by test selection
LINT_ONLY_FILES = {
    ".pre-commit-config.yaml",
    # add later if you use them:
    # ".pre-commit-hooks.yaml",
}


# Category rules are location-based. If multiple categories match, we treat it as mixed.
# For prototype simplicity:
#   - docs-only if *all* files are docs-like
#   - if >2 categories match -> FULL - FIXME: this is too blunt, find better heuristics or simply refine rules/full triggers
#   - else -> FAST with merged selections
# TODO @C-Achard Refine selection and rules
#####
# Purpose and usage:
#   - Each category has a set of matchers (predicates on file paths)
#   - If any file matches a category, that category is active
#   - Each category contributes pytest paths and/or functional scripts
# If any FULL_SUITE_TRIGGERS match, we skip categories and go straight to FULL, below rules do not apply.
# If the fast lane is selected, we take the union of all pytest paths and scripts from active categories.
CATEGORY_RULES = [
    {
        "name": "docs",
        "match_any": [
            lambda p: p.startswith("docs/"),
            lambda p: p.endswith(".md"),
            lambda p: p.endswith(".rst"),
            lambda p: p.endswith(".ipynb") and "docs" in p.lower(),
            lambda p: p in {"_config.yml", "_toc.yml"},
        ],
        "pytest_paths": [
            # NOTE: if you add tests here, the DOCS_ONLY plan will be ignored and escalated to FAST
            # Be aware of this behavior when attaching rules to the "docs" category, and prefer editing the docs workflow directly
            # in .github/workflows/build-book.yml
        ],
        "functional_scripts": [
            # NOTE: if you add scripts here, the DOCS_ONLY plan will be ignored and escalated to FAST
            # Be aware of this behavior when attaching rules to the "docs" category, and prefer editing the docs workflow directly
            # in .github/workflows/build-book.yml
        ],
    },
    {
        "name": "notebooks_examples",
        "match_any": [
            lambda p: p.startswith("examples/JUPYTER/"),
            lambda p: p.startswith("examples/COLAB/"),
            lambda p: p.endswith(".ipynb") and "examples" in p.lower(),
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
        "name": "ci_workflows",
        "match_any": [
            lambda p: p.startswith(".github/"),
            # lambda p: p.startswith("tools/"),
        ],
        "pytest_paths": MINIMAL_PYTEST,
        "functional_scripts": [],
    },
    {
        "name": "ci_tools",
        "match_any": [
            lambda p: p.startswith("tools/"),
        ],
        "pytest_paths": ["tests/tools/"],
        "functional_scripts": [],
    },
]
CATEGORY_RULE_BY_NAME: Dict[str, Dict[str, Any]] = {
    r["name"]: r for r in CATEGORY_RULES
}


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


def _empty_tree(repo: Path) -> str:
    # Avoid hardcoding; derive the empty tree hash deterministically.
    empty = _run_git(["hash-object", "-t", "tree", os.devnull], repo)
    return _validate_sha("empty-tree", empty)


def determine_diff_range(
    repo: Path, override_base: Optional[str], override_head: Optional[str]
) -> Tuple[str, str, DiffMode]:
    """Return (base_commit, head_commit, mode)."""
    zero_sha = "0" * 40
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
        return merge_base, head, DiffMode.MANUAL

    if event_name == "pull_request" and "pull_request" in event:
        base_sha = _validate_sha("base", event["pull_request"]["base"]["sha"])
        head_sha = _validate_sha("head", event["pull_request"]["head"]["sha"])
        _ensure_commit_exists(base_sha, repo)
        _ensure_commit_exists(head_sha, repo)
        # Use merge-base to approximate the PR triple-dot diff base deterministically.
        merge_base = _run_git(["merge-base", head_sha, base_sha], repo)
        merge_base = _validate_sha("merge-base", merge_base)
        _ensure_commit_exists(merge_base, repo)
        return merge_base, head_sha, DiffMode.PR

    if event_name == "push" and "before" in event and "after" in event:
        before = _validate_sha("before", event["before"])
        after = _validate_sha("after", event["after"])
        _ensure_commit_exists(after, repo)

        if before == zero_sha:
            empty = _empty_tree(repo)
            return empty, after, DiffMode.INITIAL
        try:
            _ensure_commit_exists(before, repo)
            return before, after, DiffMode.PUSH
        except Exception:
            empty = _empty_tree(repo)
            return empty, after, DiffMode.INITIAL

    # Fallback: try parent..HEAD; if no parent (initial commit), diff empty-tree..HEAD
    try:
        head = _validate_sha("HEAD", _run_git(["rev-parse", "HEAD"], repo))
        _ensure_commit_exists(head, repo)

        try:
            prev = _validate_sha(
                "HEAD^", _run_git(["rev-parse", "--verify", "HEAD^"], repo)
            )
            _ensure_commit_exists(prev, repo)
            return prev, head, DiffMode.FALLBACK
        except Exception:
            # Initial commit (no parent): treat as "everything added"
            empty = _empty_tree(repo)
            return empty, head, DiffMode.INITIAL
    except Exception:
        return "", "", DiffMode.FALLBACK_NO_HEAD


def changed_files(repo: Path, base: str, head: str) -> List[str]:
    if not base or not head:
        return []
    out = _run_git(["diff", "--name-only", "--diff-filter=ACMRTD", base, head], repo)
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
    triggered = []
    for f in files:
        for name, pred in FULL_SUITE_TRIGGERS:
            if _matches_any(f, [pred]):
                triggered.append((f, name))

    if triggered:
        reasons.append("full_suite_trigger")
        # Optional: add a compact reason count (still machine-readable)
        reasons.append(f"full_suite_trigger_count:{len(triggered)}")
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
        docs_pytests = list(docs_rule.get("pytest_paths", []) or [])
        docs_scripts = list(docs_rule.get("functional_scripts", []) or [])

        # If docs category has explicit test/script rules attached, run FAST lane.
        if docs_pytests or docs_scripts:
            return SelectorResult(
                plan=Plan.FAST,
                pytest_paths=sorted(set(docs_pytests)),
                functional_scripts=sorted(set(docs_scripts)),
                reasons=["docs_only_but_rules_attached", "category:docs"],
                changed_files=files,
            )

        # Otherwise, true docs-only (no tests)
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

    # Provenance maps
    pytest_sources: Dict[str, Set[str]] = defaultdict(set)
    script_sources: Dict[str, Set[str]] = defaultdict(set)

    for rule in matched:
        cat = rule["name"]
        reasons.append(f"category:{cat}")

        for p in rule.get("pytest_paths", []):
            pytest_paths_set.add(p)
            pytest_sources[p].add(cat)

        for s in rule.get("functional_scripts", []):
            functional_set.add(s)
            script_sources[s].add(cat)

    if not pytest_paths_set and not functional_set:
        for p in MINIMAL_PYTEST:
            pytest_paths_set.add(p)
            pytest_sources[p].add("fallback_minimal_pytest")
        reasons.append("fallback_minimal_pytest")

    res = SelectorResult(
        plan=Plan.FAST,
        pytest_paths=sorted(pytest_paths_set),
        functional_scripts=sorted(functional_set),
        reasons=reasons,
        changed_files=files,
    )

    return res


# -----------------------------
# Outputs
# -----------------------------
def explain_changed_files(files: List[str]) -> Dict[str, Any]:
    """
    Build an explanation structure for reporting:
      - per-file: full_trigger_matches, category_matches
      - grouped: full_triggers, by_category, uncategorized
    """
    per_file: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, List[str]] = defaultdict(list)
    full_trigger_files: Dict[str, List[str]] = defaultdict(list)
    lint_only_files: List[str] = []
    uncategorized: List[str] = []

    # Prep category predicates
    categories = [(r["name"], r["match_any"]) for r in CATEGORY_RULES]

    for f in files:
        # Which full-suite triggers does this file match?
        ft = []
        for trig_name, pred in FULL_SUITE_TRIGGERS:
            try:
                if pred(f):
                    ft.append(trig_name)
            except Exception:
                continue

        # Which categories does it match?
        cats = []
        for cat_name, preds in categories:
            if _matches_any(f, preds):
                cats.append(cat_name)
        is_lint_only = f in LINT_ONLY_FILES

        per_file[f] = {
            "full_triggers": ft,
            "categories": cats,
            "lint_only": is_lint_only,
        }

        if ft:
            for t in ft:
                full_trigger_files[t].append(f)

        if cats:
            for c in cats:
                by_category[c].append(f)

        if is_lint_only:
            lint_only_files.append(f)
        else:
            # Only uncategorized if it matched no categories AND no full-suite triggers
            if not ft and not cats:
                uncategorized.append(f)

    # Deterministic ordering
    for t in full_trigger_files:
        full_trigger_files[t] = sorted(set(full_trigger_files[t]))
    for c in by_category:
        by_category[c] = sorted(set(by_category[c]))

    return {
        "per_file": per_file,
        "full_trigger_files": dict(full_trigger_files),
        "by_category": dict(by_category),
        "lint_only": sorted(set(lint_only_files)),
        "uncategorized": sorted(set(uncategorized)),
    }


def _render_file_line(
    f: str,
    info: Dict[str, Any],
    emoji: bool = False,
    add_tag: bool = True,
    add_marker: bool = False,
    category_only: bool = True,
) -> str:
    # Optional, single marker only
    marker = ""
    if add_marker:
        if info.get("full_triggers"):
            marker = "⚠️ " if emoji else ""
        elif info.get("lint_only"):
            marker = "🧹 " if emoji else ""
        elif not info.get("categories"):
            marker = "❓ " if emoji else ""

    tags = []
    if add_tag:
        if info.get("categories"):
            header = "🏷️ " if emoji else "Category match :"
            tags.append(f"{header} " + ", ".join(info["categories"]))
        if info.get("full_triggers") and not category_only:
            header = "🚨 " if emoji else "Full triggers"
            tags.append(f"{header} " + ", ".join(info["full_triggers"]))
        if info.get("lint_only") and not category_only:
            header = "🧹 " if emoji else "Lint-only :"
            tags.append(f"{header}")

    tag_str = (" — " + " | ".join(tags)) if tags else ""
    return f"- {marker}`{f}`{tag_str}"


def _compute_selection_provenance(
    res: SelectorResult,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Infer which categories contributed which selected pytest/script paths.
    Returns:
      { "pytest": {path: [sources...]}, "scripts": {path: [sources...]} }
    """
    pytest_sources: Dict[str, Set[str]] = defaultdict(set)
    script_sources: Dict[str, Set[str]] = defaultdict(set)

    # Which categories are active for the current changed files?
    files = res.changed_files
    matched_rules = []
    for rule in CATEGORY_RULES:
        if any(_matches_any(f, rule["match_any"]) for f in files):
            matched_rules.append(rule)

    # Attribute sources based on matched rules
    for rule in matched_rules:
        cat = rule["name"]
        for p in rule.get("pytest_paths", []):
            if p in res.pytest_paths:
                pytest_sources[p].add(cat)
        for s in rule.get("functional_scripts", []):
            if s in res.functional_scripts:
                script_sources[s].add(cat)

    # If minimal fallback happened, it will show in reasons
    if any(r == "fallback_minimal_pytest" for r in res.reasons):
        for p in res.pytest_paths:
            if p in MINIMAL_PYTEST:
                pytest_sources[p].add("fallback_minimal_pytest")

    # Deterministic output
    return {
        "pytest": {k: sorted(v) for k, v in sorted(pytest_sources.items())},
        "scripts": {k: sorted(v) for k, v in sorted(script_sources.items())},
    }


def _compact_reasons(reasons: List[str]) -> List[str]:
    cats = sorted({r.split(":", 1)[1] for r in reasons if r.startswith("category:")})
    other = [r for r in reasons if not r.startswith("category:")]
    out = []
    if cats:
        out.append("categories: " + ", ".join(cats))
    out.extend(other)
    return out


def _render_decision_markdown(
    res: SelectorResult,
    limit: int = 40,
    style: str = "minimal",
    emoji: bool = False,
) -> str:
    def bullet(items: List[str], limit_: int = limit) -> str:
        if not items:
            return "_(none)_"
        shown = items[:limit_]
        s = "\n".join(f"- `{x}`" for x in shown)
        if len(items) > limit_:
            s += f"\n- … and {len(items) - limit_} more"
        return s

    # Plan line (minimal, no emoji by default)
    plan_label = res.plan.value
    if emoji:
        plan_label = {
            Plan.DOCS_ONLY: "📚 Documentation checks",
            Plan.FAST: "⚡ Fast, targeted tests",
            Plan.FULL: "🧪 Full suite",
        }.get(res.plan, res.plan.value)

    diff_mode = f"{MODE_LABELS.get(res.diff_mode, res.diff_mode.value)}"

    md: List[str] = []
    md.append("# Test selection\n")
    md.append(f"**Plan:** `{plan_label}`\n")
    md.append(f"**Diff mode:** `{diff_mode}`\n")

    # Reasons (compacted)
    md.append("## Why\n")
    for r in _compact_reasons(res.reasons):
        md.append(f"- `{r}`")
    md.append("")

    # Explain changed files
    exp = explain_changed_files(res.changed_files)

    md.append("## Changed files (explained)\n")

    # 1) Collapsible: Files that match full-suite triggers
    # (Always collapsible if present; otherwise omit section.)
    if exp["full_trigger_files"]:
        total_triggered = sum(len(v) for v in exp["full_trigger_files"].values())
        md.append(
            f"<details><summary><strong>Files that match full-suite triggers</strong> ({total_triggered})</summary>\n"
        )
        md.append("")
        for trig_name in sorted(exp["full_trigger_files"].keys()):
            files_for_trigger = exp["full_trigger_files"][trig_name]
            md.append(f"**{trig_name}** ({len(files_for_trigger)})")
            for f in files_for_trigger[:limit]:
                md.append(_render_file_line(f, exp["per_file"][f], emoji=emoji))
            if len(files_for_trigger) > limit:
                md.append(f"- … and {len(files_for_trigger) - limit} more")
            md.append("")
        md.append("</details>\n")

    # 2) Files grouped by category (includes uncategorized and lint-only as collapsible lists)
    md.append("### Files grouped by category\n")

    if exp["by_category"]:
        for cat in sorted(exp["by_category"].keys()):
            files = exp["by_category"][cat]

            # Determine if this category has any explicit selection rules attached
            rule = CATEGORY_RULE_BY_NAME.get(cat, {})
            has_rules = bool(rule.get("pytest_paths") or rule.get("functional_scripts"))
            note = "" if has_rules else " — no specific testing rules attached"

            md.append(
                f"<details><summary><strong>{cat}</strong> ({len(files)}){note}</summary>\n"
            )
            md.append("")
            for f in files[:limit]:
                # Already grouped by category; keep lines clean
                md.append(f"- `{f}`")
            if len(files) > limit:
                md.append(f"- … and {len(files) - limit} more")
            md.append("\n</details>\n")
    else:
        md.append("_(none)_\n")

    # Lint-only as collapsible
    if exp.get("lint_only"):
        lint_files = exp["lint_only"]
        md.append(
            f"<details><summary><strong>Lint-only</strong> ({len(lint_files)}) — ignored for test selection</summary>\n"
        )
        md.append("")
        for f in lint_files[:limit]:
            md.append(f"- `{f}`")
        if len(lint_files) > limit:
            md.append(f"- … and {len(lint_files) - limit} more")
        md.append("\n</details>\n")

    # Uncategorized as collapsible — and clarify what it means
    # IMPORTANT: explain_changed_files() already ensures that files that match ANY category
    # never land here. This section is only for truly unmatched files.
    if exp["uncategorized"]:
        unc_files = exp["uncategorized"]
        md.append(
            f"<details><summary><strong>Uncategorized</strong> ({len(unc_files)}) — no matching category (no specific testing rules attached)</summary>\n"
        )
        md.append("")
        for f in unc_files[:limit]:
            md.append(_render_file_line(f, exp["per_file"][f], emoji=emoji))
        if len(unc_files) > limit:
            md.append(f"- … and {len(unc_files) - limit} more")
        md.append("\n</details>\n")

    if style == "detailed":
        md.append("## Changed files (raw)\n")
        md.append(bullet(res.changed_files))
        md.append("")

    # Selected tests
    md.append("## Selected tests\n")
    md.append("<details><summary><strong>Pytest paths</strong></summary>\n")
    md.append(bullet(res.pytest_paths))
    md.append("\n</details>\n")
    md.append("<details><summary><strong>Functional scripts</strong></summary>\n")
    md.append(bullet(res.functional_scripts))
    md.append("\n</details>\n")

    # Provenance collapsed by default, only if detailed
    if style == "detailed":
        prov = _compute_selection_provenance(res)
        md.append("## Provenance\n")
        md.append("<details><summary><strong>Why these tests</strong></summary>\n")
        md.append("")
        if prov["pytest"]:
            md.append("### Pytest\n")
            for p, srcs in prov["pytest"].items():
                md.append(f"- `{p}` ← {', '.join(f'`{s}`' for s in srcs)}")
        else:
            md.append("### Pytest\n_(none)_")

        if prov["scripts"]:
            md.append("\n### Scripts\n")
            for s, srcs in prov["scripts"].items():
                md.append(f"- `{s}` ← {', '.join(f'`{x}`' for x in srcs)}")
        else:
            md.append("\n### Scripts\n_(none)_")

        md.append("\n</details>\n")
    return "\n".join(md)


def write_report_files(
    res: SelectorResult,
    out_dir: Path,
    report_style: str = "minimal",
    no_emoji: bool = False,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "selection.json"
    md_path = out_dir / "decision.md"

    json_path.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(
        _render_decision_markdown(res, style=report_style, emoji=not no_emoji),
        encoding="utf-8",
    )
    return json_path, md_path


def create_job_summary(md_path: Path, overwrite: bool = True) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    # Append markdown to the GitHub Actions Job Summary
    mode = "w" if overwrite else "a"
    with open(summary_path, mode, encoding="utf-8") as f:
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
        f.write(f"diff_mode={res.diff_mode.value}\n")
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
    ap.add_argument(
        "--report-style",
        choices=["minimal", "detailed"],
        default="minimal",
        help="Decision markdown verbosity: minimal (default) or detailed",
    )
    ap.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emojis in markdown report (default: off)",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    repo = find_repo_root()

    base, head, diff_mode = determine_diff_range(repo, args.base_sha, args.head_sha)
    files = changed_files(repo, base, head)

    res = decide(files)
    res.diff_mode = diff_mode
    res.changed_files = files
    res = validate_selected_paths(res, repo)

    # Strict validation
    try:
        res = SelectorResult.model_validate(res.model_dump())
    except ValidationError as e:
        raise RuntimeError(f"Output validation failed: {e}") from e

    # Always write report files for transparency
    report_dir = Path(args.report_dir)
    json_path, md_path = write_report_files(
        res, report_dir, report_style=args.report_style, no_emoji=args.no_emoji
    )

    if args.json:
        print(res.model_dump_json(indent=2))

    if args.write_github_output:
        write_github_output(res)

    # Write Job Summary (GitHub renders markdown from $GITHUB_STEP_SUMMARY)
    if args.write_summary:
        create_job_summary(md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
