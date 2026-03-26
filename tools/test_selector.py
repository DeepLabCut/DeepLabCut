#!/usr/bin/env python3
"""Deterministic, strictly validated test selector for DeepLabCut.

Outputs orthogonal workflow mode selections plus structured test selections:

  - lanes: which workflow lanes should run (skip, docs, fast, full)
  - pytest_paths: JSON list of pytest path arguments
  - functional_scripts: JSON list of python script paths
  - provenance: JSON mapping each selected test/script to the category rules that selected it

Safety principles
-----------------
- Fail-safe: if changes cannot be determined or are ambiguous, the "full" lane is always selected.
- Deterministic: derives diff range from GitHub Actions event payload when available.
  * pull_request: uses merge-base(base.sha, head.sha) .. head.sha
  * push: uses before .. after
  * manual override: uses exactly --base-sha .. --head-sha
  * fallback: attempts HEAD~1 .. HEAD
- Secure: never emits shell command strings; only structured data.
- Strict: Pydantic schema validation (extra=forbid), SHA validation, path sanitization.

Intended usage in GitHub Actions
-------------------------------
- Checkout with sufficient history for merge-base/diff (typically fetch-depth: 0).
- Run:
    python tools/test_selector.py --write-github-output --json

This will write the following keys to $GITHUB_OUTPUT:
    - run_skip (bool): whether to run the skip mode
    - run_docs (bool): whether to run the docs workflow
    - run_fast (bool): whether to run targeted test execution
    - run_full (bool): whether to run the full matrix/full suite workflow
    - selected_workflows (list): list of selected workflow lanes
    - lane_reasons (dict): reasons for selecting each workflow lane
    - diff_mode (str): how the diff was determined
    - pytest_paths (list): list of pytest path arguments
    - functional_scripts (list): list of python script paths
    - reasons (list): aggregate machine-readable reasons for the selection
    - changed_files (list): list of changed files
    - provenance (dict): mapping each selected test/script to the category rules that selected it

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
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    from .test_selector_config import (
        CATEGORY_RULE_BY_NAME,
        CATEGORY_RULES,
        FULL_SUITE_TRIGGERS,
        LINT_ONLY_FILES,
        MINIMAL_PYTEST,
    )
# Allows to run as "python tools/test_selector.py" without installing as a package,
# but still import the config from the same location.
except ImportError:  # pragma: no cover
    from test_selector_config import (
        CATEGORY_RULE_BY_NAME,
        CATEGORY_RULES,
        FULL_SUITE_TRIGGERS,
        LINT_ONLY_FILES,
        MINIMAL_PYTEST,
    )


SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
# DLC_NAMESPACE = "deeplabcut"


class DiffMode(str, Enum):
    """How the diff was determined, for auditing and reporting."""

    PR = "pr"  # merge-base(base, head) .. head
    PUSH = "push"  # before .. after
    MANUAL = "manual"  # explicit base...head from CLI args
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


class LaneSelection(BaseModel):
    """Which workflow lanes should run."""

    model_config = ConfigDict(extra="forbid")

    skip: bool = False  # Skip all tests (e.g. lint-only changes only)
    docs: bool = False  # Run docs build checks
    fast: bool = False  # Run targeted pytest + optional functional scripts in test workflow
    full: bool = False  # Delegate to full test workflow/matrix


class SelectionProvenance(BaseModel):
    """Why each selected test/script path was included."""

    model_config = ConfigDict(extra="forbid")

    pytest: dict[str, list[str]] = Field(default_factory=dict)
    scripts: dict[str, list[str]] = Field(default_factory=dict)


class SelectorResult(BaseModel):
    """Strict output schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    diff_mode: DiffMode = DiffMode.FALLBACK_NO_HEAD

    lanes: LaneSelection = Field(default_factory=LaneSelection)

    pytest_paths: list[str] = Field(default_factory=list)
    functional_scripts: list[str] = Field(default_factory=list)
    provenance: SelectionProvenance = Field(default_factory=SelectionProvenance)

    reasons: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    lane_reasons: dict[str, list[str]] = Field(default_factory=dict)


SelectorResult.model_rebuild()  # Ensure model is fully built at import time for validation in main()


# -----------------------------
# Git helpers
# -----------------------------
def _run_git(args: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
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


def _load_github_event() -> dict[str, Any]:
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


def determine_diff_range(repo: Path, override_base: str | None, override_head: str | None) -> tuple[str, str, DiffMode]:
    """Return (base_commit, head_commit, mode)."""
    zero_sha = "0" * 40
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = _load_github_event()

    if override_base and override_head:
        base = _validate_sha("base", override_base)
        head = _validate_sha("head", override_head)
        _ensure_commit_exists(base, repo)
        _ensure_commit_exists(head, repo)
        return base, head, DiffMode.MANUAL

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
            prev = _validate_sha("HEAD^", _run_git(["rev-parse", "--verify", "HEAD^"], repo))
            _ensure_commit_exists(prev, repo)
            return prev, head, DiffMode.FALLBACK
        except Exception:
            # Initial commit (no parent): treat as "everything added"
            empty = _empty_tree(repo)
            return empty, head, DiffMode.INITIAL
    except Exception:
        return "", "", DiffMode.FALLBACK_NO_HEAD


def changed_files(repo: Path, base: str, head: str) -> list[str]:
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
    missing: list[str] = []

    # validate pytest paths (files/dirs)
    for p in res.pytest_paths:
        if not _is_safe_relpath(p) or not (repo / p).exists():
            missing.append(f"pytest:{p}")

    # validate functional scripts (files)
    for s in res.functional_scripts:
        if not _is_safe_relpath(s) or not (repo / s).exists():
            missing.append(f"script:{s}")

    if missing:
        # Fail-safe escalation: disable fast lane selection, enable full lane.
        # Preserve docs lane if it was independently selected.
        res.lanes.fast = False
        res.lanes.full = True
        res.pytest_paths = []
        res.functional_scripts = []
        res.provenance = SelectionProvenance()
        res.reasons = res.reasons + ["missing_selected_paths"] + missing

        lane_reasons = dict(res.lane_reasons)
        lane_reasons.pop("fast", None)
        full_reasons = list(lane_reasons.get("full", []))
        full_reasons.extend(["missing_selected_paths", *missing])
        lane_reasons["full"] = full_reasons
        res.lane_reasons = lane_reasons

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


def decide(files: list[str]) -> SelectorResult:
    reasons: list[str] = []
    lane_reasons: dict[str, list[str]] = {}
    lanes = LaneSelection()

    if not files:
        lanes.full = True
        full_reasons = ["no_changed_files_or_diff_unavailable"]
        return SelectorResult(
            lanes=lanes,
            pytest_paths=[],
            functional_scripts=[],
            provenance=SelectionProvenance(),
            reasons=full_reasons,
            changed_files=[],
            lane_reasons={"full": full_reasons},
        )

    # Lint-only filtering (routing should ignore these files)
    lint_only = [f for f in files if f in LINT_ONLY_FILES]
    routed_files = [f for f in files if f not in LINT_ONLY_FILES]

    if lint_only:
        reasons.append(f"lint_only_count:{len(lint_only)}")

    # If *only* lint-only files changed, skip all lanes
    if not routed_files:
        lanes.skip = True
        skip_reasons = [*reasons, "lint_only"]
        return SelectorResult(
            lanes=lanes,
            pytest_paths=[],
            functional_scripts=[],
            provenance=SelectionProvenance(),
            reasons=skip_reasons,
            changed_files=files,
            lane_reasons={"skip": skip_reasons},
        )

    # Docs lane is orthogonal: if any routed file matches docs, enable docs lane.
    docs_rule = CATEGORY_RULE_BY_NAME.get("docs")
    docs_touched = bool(docs_rule and any(_matches_any(f, docs_rule.match_any) for f in routed_files))
    docs_matched_files = {f for f in routed_files if docs_rule and _matches_any(f, docs_rule.match_any)}
    non_docs_routed_files = [f for f in routed_files if f not in docs_matched_files]

    docs_pytests_sorted: list[str] = []
    docs_scripts_sorted: list[str] = []

    if docs_touched:
        lanes.docs = True
        reasons.append("category:docs")
        lane_reasons["docs"] = ["category:docs"]
        docs_pytests_sorted = sorted(set(docs_rule.pytest_paths)) if docs_rule else []
        docs_scripts_sorted = sorted(set(docs_rule.functional_scripts)) if docs_rule else []

    # Full-suite triggers always win over fast, but docs lane can still remain enabled.
    triggered: list[tuple[str, str]] = []
    for f in routed_files:
        for name, pred in FULL_SUITE_TRIGGERS:
            if _matches_any(f, [pred]):
                triggered.append((f, name))

    if triggered:
        lanes.full = True
        full_reasons = [
            "full_suite_trigger",
            f"full_suite_trigger_count:{len(triggered)}",
        ]
        reasons.extend(full_reasons)
        lane_reasons["full"] = full_reasons
        return SelectorResult(
            lanes=lanes,
            pytest_paths=[],
            functional_scripts=[],
            provenance=SelectionProvenance(),
            reasons=reasons,
            changed_files=files,
            lane_reasons=lane_reasons,
        )

    # Match NON-doc categories only for test-routing / escalation logic.
    matched_non_docs = []
    for rule in CATEGORY_RULES:
        if rule.name == "docs":
            continue
        if any(_matches_any(f, rule.match_any) for f in routed_files):
            matched_non_docs.append(rule)

    matched_non_docs = sorted(matched_non_docs, key=lambda r: r.name)

    for rule in matched_non_docs:
        reasons.append(f"category:{rule.name}")

    pytest_paths_set: set[str] = set()
    functional_set: set[str] = set()
    pytest_sources: dict[str, set[str]] = defaultdict(set)
    script_sources: dict[str, set[str]] = defaultdict(set)

    # Docs rules may contribute tests/scripts to the fast lane.
    if docs_touched:
        for p in docs_pytests_sorted:
            pytest_paths_set.add(p)
            pytest_sources[p].add("docs")
        for s in docs_scripts_sorted:
            functional_set.add(s)
            script_sources[s].add("docs")

    # Non-doc matched categories contribute to fast lane.
    for rule in matched_non_docs:
        cat = rule.name
        for p in rule.pytest_paths or []:
            pytest_paths_set.add(p)
            pytest_sources[p].add(cat)
        for s in rule.functional_scripts or []:
            functional_set.add(s)
            script_sources[s].add(cat)

    # If we matched non-doc categories but none provided explicit tests/scripts,
    # fall back to the minimal pytest lane.
    fallback_used = False
    if not pytest_paths_set and not functional_set and matched_non_docs:
        for p in MINIMAL_PYTEST:
            pytest_paths_set.add(p)
            pytest_sources[p].add("fallback_minimal_pytest")
        reasons.append("fallback_minimal_pytest")
        fallback_used = True

    # If the routed changes are truly docs-only (no non-doc files remain) and no
    # tests were selected, return docs lane only.
    if not pytest_paths_set and not functional_set:
        if lanes.docs and not non_docs_routed_files:
            return SelectorResult(
                lanes=lanes,
                pytest_paths=[],
                functional_scripts=[],
                provenance=SelectionProvenance(),
                reasons=reasons,
                changed_files=files,
                lane_reasons=lane_reasons,
            )

        # Otherwise fail-safe to full when nothing matched at all.
        lanes.full = True
        full_reasons = ["no_category_matched"]
        reasons.extend(full_reasons)
        lane_reasons["full"] = full_reasons
        return SelectorResult(
            lanes=lanes,
            pytest_paths=[],
            functional_scripts=[],
            provenance=SelectionProvenance(),
            reasons=reasons,
            changed_files=files,
            lane_reasons=lane_reasons,
        )

    # Fast lane selected
    lanes.fast = True

    fast_reasons: list[str] = []
    if docs_touched and (docs_pytests_sorted or docs_scripts_sorted):
        fast_reasons.append("category:docs")
    fast_reasons.extend(f"category:{rule.name}" for rule in matched_non_docs)
    if fallback_used:
        fast_reasons.append("fallback_minimal_pytest")
    lane_reasons["fast"] = fast_reasons

    return SelectorResult(
        lanes=lanes,
        pytest_paths=sorted(pytest_paths_set),
        functional_scripts=sorted(functional_set),
        provenance=SelectionProvenance(
            pytest={k: sorted(v) for k, v in sorted(pytest_sources.items())},
            scripts={k: sorted(v) for k, v in sorted(script_sources.items())},
        ),
        reasons=reasons,
        changed_files=files,
        lane_reasons=lane_reasons,
    )


# -----------------------------
# Outputs
# -----------------------------
def explain_changed_files(files: list[str]) -> dict[str, Any]:
    """
    Build an explanation structure for reporting:
      - per-file: full_trigger_matches, category_matches
      - grouped: full_triggers, by_category, uncategorized
    """
    per_file: dict[str, dict[str, Any]] = {}
    by_category: dict[str, list[str]] = defaultdict(list)
    full_trigger_files: dict[str, list[str]] = defaultdict(list)
    lint_only_files: list[str] = []
    uncategorized: list[str] = []

    # Prep category predicates
    categories = [(r.name, r.match_any) for r in CATEGORY_RULES]

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
    info: dict[str, Any],
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


def _enabled_lane_names(res: SelectorResult) -> list[str]:
    order = ("skip", "docs", "fast", "full")
    return [name for name in order if getattr(res.lanes, name)]


def _lane_label(name: str, emoji: bool = False) -> str:
    if not emoji:
        return name
    return {
        "skip": "⏩ skip",
        "docs": "📚 docs",
        "fast": "⚡ fast",
        "full": "🧪 full",
    }.get(name, name)


def _compact_reasons(reasons: list[str]) -> list[str]:
    cats = sorted({r.split(":", 1)[1] for r in reasons if r.startswith("category:")})
    other = [r for r in reasons if not r.startswith("category:")]
    out = []
    if cats:
        out.append("categories: " + ", ".join(cats))
    out.extend(other)
    return out


def _details_open(summary: str, add_blank: bool = True) -> str:
    s = f"<details><summary><strong>{summary}</strong></summary>\n"
    if add_blank:
        s += "\n"
    return s


def _details_close() -> str:
    return "\n</details>\n"


def _render_decision_markdown(
    res: SelectorResult,
    limit: int = 40,
    style: str = "minimal",
    emoji: bool = False,
) -> str:
    def bullet(items: list[str], limit_: int = limit) -> str:
        if not items:
            return "_(none)_"
        shown = items[:limit_]
        s = "\n".join(f"- `{x}`" for x in shown)
        if len(items) > limit_:
            s += f"\n- … and {len(items) - limit_} more"
        return s

    # Selection line (minimal, no emoji by default)
    selected_lanes = _enabled_lane_names(res)
    if emoji:
        selected_lanes_label = ", ".join(_lane_label(name, emoji=True) for name in selected_lanes)
    else:
        selected_lanes_label = ", ".join(f"`{name}`" for name in selected_lanes)

    if not selected_lanes_label:
        selected_lanes_label = "_(none)_"

    diff_mode = f"{MODE_LABELS.get(res.diff_mode, res.diff_mode.value)}"

    md: list[str] = []
    md.append("# Test selection\n")
    md.append(f"**Selected workflows:** {selected_lanes_label}\n")
    md.append(f"**Diff mode:** `{diff_mode}`\n")

    # Reasons (compacted)
    md.append("## Why\n")
    for r in _compact_reasons(res.reasons):
        md.append(f"- `{r}`")
    md.append("")

    if style == "detailed" and res.lane_reasons:
        md.append("## Workflow lanes\n")
        for lane in _enabled_lane_names(res):
            lane_rs = res.lane_reasons.get(lane, [])
            md.append(f"### `{lane}`")
            if lane_rs:
                for r in lane_rs:
                    md.append(f"- `{r}`")
            else:
                md.append("_(none)_")
            md.append("")

    # Explain changed files
    exp = explain_changed_files(res.changed_files)

    md.append("## Changed files (explained)\n")

    # 1) Collapsible: Files that match full-suite triggers
    # (Always collapsible if present; otherwise omit section.)
    if exp["full_trigger_files"]:
        total_triggered = sum(len(v) for v in exp["full_trigger_files"].values())
        md.append(_details_open(f"Files that match full-suite triggers ({total_triggered})"))
        for trig_name in sorted(exp["full_trigger_files"].keys()):
            files_for_trigger = exp["full_trigger_files"][trig_name]
            md.append(f"**{trig_name}** ({len(files_for_trigger)})")
            for f in files_for_trigger[:limit]:
                md.append(_render_file_line(f, exp["per_file"][f], emoji=emoji))
            if len(files_for_trigger) > limit:
                md.append(f"- … and {len(files_for_trigger) - limit} more")
            md.append("")
        md.append(_details_close())

    # 2) Files grouped by category (includes uncategorized and lint-only as collapsible lists)
    md.append("### Files grouped by category\n")

    if exp["by_category"]:
        for cat in sorted(exp["by_category"].keys()):
            files = exp["by_category"][cat]

            # Determine if this category has any explicit selection rules attached
            rule = CATEGORY_RULE_BY_NAME.get(cat)
            has_rules = bool(rule and (rule.pytest_paths or rule.functional_scripts))
            note = "" if has_rules else " — no specific testing rules attached"

            md.append(_details_open(f"{cat} ({len(files)}){note}"))
            for f in files[:limit]:
                # Already grouped by category; keep lines clean
                md.append(f"- `{f}`")
            if len(files) > limit:
                md.append(f"- … and {len(files) - limit} more")
            md.append(_details_close())
    else:
        md.append("_(none)_\n")

    # Lint-only as collapsible
    if exp.get("lint_only"):
        lint_files = exp["lint_only"]
        md.append(_details_open(f"Lint-only ({len(lint_files)}) — ignored for test selection"))
        md.append("")
        for f in lint_files[:limit]:
            md.append(f"- `{f}`")
        if len(lint_files) > limit:
            md.append(f"- … and {len(lint_files) - limit} more")
        md.append(_details_close())

    # Uncategorized as collapsible — and clarify what it means
    # IMPORTANT: explain_changed_files() already ensures that files that match ANY category
    # never land here. This section is only for truly unmatched files.
    if exp["uncategorized"]:
        unc_files = exp["uncategorized"]
        md.append(
            _details_open(
                f"Uncategorized ({len(unc_files)}) — no matching category (no specific testing rules attached)"
            )
        )
        md.append("")
        for f in unc_files[:limit]:
            md.append(_render_file_line(f, exp["per_file"][f], emoji=emoji))
        if len(unc_files) > limit:
            md.append(f"- … and {len(unc_files) - limit} more")
        md.append(_details_close())

    if style == "detailed":
        md.append("## Changed files (raw)\n")
        md.append(bullet(res.changed_files))
        md.append("")

    # Selected tests
    md.append("## Selected tests\n")
    md.append(_details_open("Pytest paths"))
    md.append(bullet(res.pytest_paths))
    md.append(_details_close())
    md.append(_details_open("Functional scripts"))
    md.append(bullet(res.functional_scripts))
    md.append(_details_close())

    # Provenance collapsed by default, only if detailed
    if style == "detailed":
        md.append("## Provenance\n")
        md.append(_details_open("Why these tests"))
        md.append("")

        if res.provenance.pytest:
            md.append("### Pytest\n")
            for p, srcs in res.provenance.pytest.items():
                md.append(f"- `{p}` ← {', '.join(f'`{s}`' for s in srcs)}")
        else:
            md.append("### Pytest\n_(none)_")

        if res.provenance.scripts:
            md.append("\n### Scripts\n")
            for s, srcs in res.provenance.scripts.items():
                md.append(f"- `{s}` ← {', '.join(f'`{x}`' for x in srcs)}")
        else:
            md.append("\n### Scripts\n_(none)_")

        md.append(_details_close())

    return "\n".join(md)


def write_report_files(
    res: SelectorResult,
    out_dir: Path,
    report_style: str = "minimal",
    no_emoji: bool = False,
) -> tuple[Path, Path]:
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

    selected_workflows = _enabled_lane_names(res)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"run_skip={str(res.lanes.skip).lower()}\n")
        f.write(f"run_docs={str(res.lanes.docs).lower()}\n")
        f.write(f"run_fast={str(res.lanes.fast).lower()}\n")
        f.write(f"run_full={str(res.lanes.full).lower()}\n")
        f.write(f"selected_workflows={j(selected_workflows)}\n")
        f.write(f"lane_reasons={j(res.lane_reasons)}\n")
        f.write(f"diff_mode={res.diff_mode.value}\n")
        f.write(f"pytest_paths={j(res.pytest_paths)}\n")
        f.write(f"functional_scripts={j(res.functional_scripts)}\n")
        f.write(f"reasons={j(res.reasons)}\n")
        f.write(f"changed_files={j(res.changed_files)}\n")
        f.write(f"provenance={j(res.provenance.model_dump())}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic DeepLabCut test selector")
    ap.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    ap.add_argument(
        "--write-github-output",
        action="store_true",
        help="Write outputs to $GITHUB_OUTPUT",
    )
    ap.add_argument(
        "--base-sha",
        default=None,
        help="Override base commit SHA for manual diff selection (must be used with --head-sha)",
    )
    ap.add_argument(
        "--head-sha",
        default=None,
        help="Override head commit SHA for manual diff selection (must be used with --base-sha)",
    )

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
        default="detailed",
        help="Decision markdown verbosity: minimal or detailed (default: detailed)",
    )
    ap.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emojis in markdown report (default: off)",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)
    if bool(args.base_sha) != bool(args.head_sha):
        ap.error("Both --base-sha and --head-sha must be provided together")

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
    json_path, md_path = write_report_files(res, report_dir, report_style=args.report_style, no_emoji=args.no_emoji)

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
