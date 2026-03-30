"""DeepLabCut docs & notebooks automated checks tool.

Goals
-----
- SAFE by default: read-only operations in CI (report/check).
- Idempotent updates (update mode) that only touch:
    * Notebook-level metadata for .ipynb (never cells/outputs)
    * YAML frontmatter for .md docs (optional)
- Uses pydantic schemas with explicit schema_version for validation.
- Aims to be contributor-friendly by default: check mode enforces configured policy, while
  surfacing scan/parsing issues without failing unless strict mode is enabled.

Terminology
-----------
last_content_updated
    Computed from git history, excluding metadata-only commits.
    (Metadata commits must include META_COMMIT_MARKER in the commit message.)

last_verified
    Human-controlled date indicating the file was verified to work/be accurate.

verified_for
    Human-controlled string, typically the project version (e.g. 3.0.0rc13).

tier
    Optional classification (left unset by default; do not auto-populate).

Usage modes
-----------
Report (read-only):
    python tools/docs_and_notebooks_check.py report

Check (read-only; policy enforcement):
    python tools/docs_and_notebooks_check.py check

    Runs scans and evaluates configured policy rules.
    Exits non-zero for policy violations.
    Scan/parsing errors are always reported in console / JSON / Markdown output,
    but are non-fatal by default unless strict mode is enabled or they imply a
    policy violation.

Update content-date field from git (write mode; requires --write):
    python tools/docs_and_notebooks_check.py update --write --set-content-date-from-git

Update verification fields for selected targets (write mode):
    python tools/docs_and_notebooks_check.py update --write --targets docs/page.md \
        --set-last-verified today --set-verified-for 3.0.0rc13

Normalize notebooks deterministically (explicit churn; write mode):
    python tools/docs_and_notebooks_check.py normalize --write --targets docs/notebook.ipynb


Configuration
-------------
Uses tools/docs_and_notebooks_report_config.yml by default.

Outputs
-------
- docs_nb_checks.json: machine-readable report
- docs_nb_checks.md: human-readable summary

Notes for CI
------------
- Ensure actions/checkout uses fetch-depth: 0 (or sufficiently deep),
  otherwise git log may not see history.
- Requires:
  - pydantic>=2,<3
  - PyYAML
  - nbformat>=5
  to be installed in the environment.
  Recommended : install in CI job directly (pip install pydantic pyyaml nbformat)
  rather than adding to requirements, since these are only needed for this tool.
"""

# tools/docs_and_notebooks_check.py
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

import nbformat
import yaml
from nbformat.validator import NotebookValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1
DLC_NAMESPACE = "deeplabcut"
OUTPUT_FILENAME = "docs_nb_checks"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = SCRIPT_DIR / "docs_and_notebooks_report_config.yml"


# -----------------------------
# Metadata commit marker / guidance
# -----------------------------
# IMPORTANT:
#   Metadata-only updates and notebook normalization rewrite files and will change
#   "git last touched" timestamps. To preserve meaningful "content age", all such
#   commits must include this marker in the commit message.
META_COMMIT_MARKER = "chore(metadata)"
SUGGESTED_TAGGED_COMMIT = f"{META_COMMIT_MARKER}: update docs/notebooks metadata"


# -----------------------------
# Pydantic schemas
# -----------------------------


class DLCMeta(BaseModel):
    """Metadata embedded in files under the `deeplabcut` namespace."""

    model_config = ConfigDict(extra="allow")

    # Tool-managed: last meaningful content update date (excluding metadata commits)
    last_content_updated: date | None = None

    # Optional tool-managed: last time metadata/normalization was performed
    last_metadata_updated: date | None = None
    # Optional human-managed verification fields
    last_verified: date | None = None
    # Version or other string indicating what this file was verified for (e.g. "3.0.0rc13")
    verified_for: str | None = None
    # Extra metadata fields for later usage (e.g. allowlist tier classification), but not currently used by the tool
    tier: str | None = None
    ignore: bool = False
    notes: str | None = None


class ScanConfig(BaseModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    warn_if_content_older_than_days: int = 365
    warn_if_verified_older_than_days: int = 365
    missing_last_verified_is_warning: bool = True

    # Strict-mode toggle: if true, scan/parsing errors also fail `check`
    fail_on_scan_errors: bool = False

    # Allowlists for strict checks (start empty; ratchet later)
    require_metadata: list[str] = Field(default_factory=list)
    require_recent_verification: list[str] = Field(default_factory=list)

    require_notebook_normalized: list[str] = Field(default_factory=list)


class ToolConfig(BaseModel):
    version: int = 1
    scan: ScanConfig
    policy: PolicyConfig


class FileRecord(BaseModel):
    path: str
    kind: str  # ipynb | md | other

    # Computed from git (excluding metadata-only commits)
    last_content_updated: date | None = None
    # Debug-only: raw git last touched (may be metadata commit)
    last_git_touched: date | None = None

    # Read from file metadata/frontmatter
    meta: DLCMeta | None = None

    # Derived
    days_since_content_update: int | None = None
    days_since_verified: int | None = None

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # If update mode would change file
    would_change: bool = False


class Report(BaseModel):
    schema_version: int = SCHEMA_VERSION
    generated_at: datetime
    repo_root: str
    config_path: str

    totals: dict[str, int]
    records: list[FileRecord]


# Rebuild models due to __future__ annotations
DLCMeta.model_rebuild()
ScanConfig.model_rebuild()
PolicyConfig.model_rebuild()
ToolConfig.model_rebuild()
FileRecord.model_rebuild()
Report.model_rebuild()
# -----------------------------
# Helpers
# -----------------------------


def _iso_today() -> date:
    return datetime.now(timezone.utc).date()


def _run_git(args: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    code, out, _err = _run_git(["rev-parse", "--show-toplevel"], cwd=start)
    if code == 0 and out:
        return Path(out).resolve()
    raise RuntimeError("Could not locate repository root")


def glob_paths(repo_root: Path, patterns: list[str]) -> list[Path]:
    results: list[Path] = []
    for pat in patterns:
        results.extend(repo_root.glob(pat))
    return sorted({p.resolve() for p in results if p.is_file()})


def is_excluded(rel_path: str, exclude_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_patterns)


def file_kind(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".ipynb":
        return "ipynb"
    if s in {".md", ".markdown"}:
        return "md"
    return "other"


def _parse_git_iso_date(out: str) -> date | None:
    out = (out or "").strip()
    if not out:
        return None

    try:
        return date.fromisoformat(out)
    except Exception:
        pass

    try:
        if out.endswith("Z"):
            out = out[:-1] + "+00:00"
        return datetime.fromisoformat(out).date()
    except Exception:
        return None


def _git_log_date(repo_root: Path, rel_path: str, extra_args: Sequence[str] = ()) -> date | None:
    args = [
        "log",
        "-1",
        "--date=short",
        "--format=%cd",
        *extra_args,
        "--",
        rel_path,
    ]
    code, out, _err = _run_git(args, cwd=repo_root)
    if code != 0:
        return None
    return _parse_git_iso_date(out)


def git_last_touched(repo_root: Path, rel_path: str) -> date | None:
    return _git_log_date(repo_root, rel_path)


def git_last_content_updated(repo_root: Path, rel_path: str) -> tuple[date | None, bool]:
    d = _git_log_date(
        repo_root,
        rel_path,
        extra_args=[
            "--fixed-strings",
            "--invert-grep",
            "--grep",
            META_COMMIT_MARKER,
        ],
    )
    if d is not None:
        return d, False
    return git_last_touched(repo_root, rel_path), True


FRONTMATTER_RE = re.compile(r"^---\s*$")


def read_md_frontmatter(text: str) -> tuple[dict | None, str, str | None]:
    lines = text.splitlines(keepends=True)
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return None, text, None

    end_idx = None
    for i in range(1, min(len(lines), 5000)):
        if FRONTMATTER_RE.match(lines[i]):
            end_idx = i
            break

    if end_idx is None:
        return None, text, "unterminated_markdown_frontmatter"

    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])

    if yaml is None:
        raise RuntimeError("PyYAML is required to parse Markdown frontmatter")

    fm = yaml.safe_load(fm_text) if fm_text.strip() else {}
    if not isinstance(fm, dict):
        return None, text, "markdown_frontmatter_not_mapping"

    return fm, body, None


def dump_md_frontmatter(frontmatter: dict, body: str) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write Markdown frontmatter")
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    body_to_write = body
    if body_to_write.startswith("\n"):
        body_to_write = body_to_write[1:]
    return "---\n" + fm_text + "---\n" + body_to_write


def read_ipynb_meta(path: Path) -> tuple[Any, dict, bool]:
    """
    Read a notebook using nbformat.
    Returns (notebook_node, deeplabcut_meta_dict, has_dlc_namespace).
    """
    nb = nbformat.read(str(path), as_version=4)

    meta = getattr(nb, "metadata", {}) or {}
    has_dlc = DLC_NAMESPACE in meta

    raw_dlc_meta = meta.get(DLC_NAMESPACE)
    return nb, raw_dlc_meta, has_dlc


def notebook_is_normalized(path: Path, nb: Any) -> bool:
    original = path.read_text(encoding="utf-8")
    # Normalize newline style so CRLF vs LF differences do not cause false mismatches
    original_normalized = original.replace("\r\n", "\n").replace("\r", "\n")
    normalized = nbformat.writes(nb, version=4, indent=2, ensure_ascii=False) + "\n"
    return original_normalized == normalized


def write_ipynb_meta(path: Path, nb: Any) -> None:
    """
    Write a notebook using nbformat.

    Note: nbformat writes JSON in a canonical form; it *will* rewrite the file,
    so expect diffs if the notebook wasn't previously normalized to the same style.
    """
    # Validate before writing (optional but recommended)
    nbformat.validate(nb)

    # Use a stable indentation to reduce churn (choose 2 if your repo tends that way)
    text = nbformat.writes(nb, version=4, indent=2, ensure_ascii=False)

    path.write_text(text + "\n", encoding="utf-8")


def parse_dlc_meta(raw: Any) -> tuple[DLCMeta | None, bool]:
    # returns (meta, valid)
    if raw is None or not isinstance(raw, dict):
        return None, False
    try:
        return DLCMeta.model_validate(raw), True
    except ValidationError:
        return None, False


def meta_to_jsonable(meta: DLCMeta) -> dict:
    """
    Return JSON-serializable metadata (dates become ISO strings).
    This prevents json.dumps() from failing when writing .ipynb files.
    """
    return meta.model_dump(mode="json", exclude_none=True)


def compute_days_since(d: date | None, today: date) -> int | None:
    return None if d is None else (today - d).days


def match_allowlist(rel_path: str, allowlist: list[str]) -> bool:
    # Support exact matches or glob patterns
    return any(pat == rel_path or fnmatch.fnmatch(rel_path, pat) for pat in allowlist)


# -----------------------------
# Core scanning
# -----------------------------


def load_config(config_path: Path) -> ToolConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ToolConfig.model_validate(raw)


def scan_files(repo_root: Path, cfg: ToolConfig, targets: list[str] | None = None) -> list[FileRecord]:
    today = _iso_today()
    paths = glob_paths(repo_root, cfg.scan.include)
    records: list[FileRecord] = []
    target_set = None
    if targets:
        target_set = set(t.replace(os.sep, "/") for t in targets)

    for p in paths:
        rel = str(p.resolve().relative_to(repo_root)).replace(os.sep, "/")
        if is_excluded(rel, cfg.scan.exclude):
            continue
        if target_set is not None and rel not in target_set:
            continue
        kind = file_kind(p)
        rec = FileRecord(path=rel, kind=kind)

        rec.last_git_touched = git_last_touched(repo_root, rel)
        rec.last_content_updated, used_fallback = git_last_content_updated(repo_root, rel)
        rec.days_since_content_update = compute_days_since(rec.last_content_updated, today)
        if used_fallback:
            rec.warnings.append("content_date_fallback_to_git_touched")

        try:
            if kind == "ipynb":
                nb, raw_meta, has_dlc = read_ipynb_meta(p)

                try:
                    nbformat.validate(nb)
                except NotebookValidationError as e:
                    rec.errors.append(f"nbformat_invalid: {e}")

                try:
                    if not notebook_is_normalized(p, nb):
                        rec.warnings.append("notebook_not_normalized")
                except Exception as e:
                    rec.errors.append(f"notebook_normalization_check_failed: {e}")

                if not has_dlc:
                    rec.meta = None
                    rec.warnings.append("missing_metadata")
                else:
                    rec.meta, valid = parse_dlc_meta(raw_meta)
                    if not valid:
                        rec.meta = None
                        rec.warnings.append("invalid_metadata")

            elif kind == "md":
                text = p.read_text(encoding="utf-8")
                fm, _body, fm_error = read_md_frontmatter(text)

                if fm_error:
                    rec.meta = None
                    rec.warnings.append("invalid_metadata")
                    rec.errors.append(f"markdown_frontmatter_invalid: {fm_error}")
                else:
                    fm = fm or {}
                    has_dlc = DLC_NAMESPACE in fm
                    raw = fm.get(DLC_NAMESPACE)

                    if not has_dlc:
                        rec.meta = None
                        rec.warnings.append("missing_metadata")
                    else:
                        rec.meta, valid = parse_dlc_meta(raw)
                        if not valid:
                            rec.meta = None
                            rec.warnings.append("invalid_metadata")

            else:
                rec.meta = None

        except Exception as e:
            rec.errors.append(f"metadata_read_failed: {e}")

        # ignore=True means: keep reporting diagnostics, but skip freshness/policy logic
        if rec.meta and rec.meta.ignore:
            records.append(rec)
            continue

        last_verified = rec.meta.last_verified if rec.meta else None
        rec.days_since_verified = compute_days_since(last_verified, today)

        # Future dates are data errors
        if rec.last_content_updated is not None and rec.last_content_updated > today:
            rec.errors.append("future_last_content_updated")
        if rec.meta and rec.meta.last_metadata_updated is not None:
            if rec.meta.last_metadata_updated > today:
                rec.errors.append("future_last_metadata_updated")
        if last_verified is not None and last_verified > today:
            rec.errors.append("future_last_verified")

        pol = cfg.policy

        if (
            rec.days_since_content_update is not None
            and rec.days_since_content_update > pol.warn_if_content_older_than_days
        ):
            rec.warnings.append(f"content_stale>{pol.warn_if_content_older_than_days}d")

        if last_verified is None and pol.missing_last_verified_is_warning:
            rec.warnings.append("missing_last_verified")
        elif rec.days_since_verified is not None and rec.days_since_verified > pol.warn_if_verified_older_than_days:
            rec.warnings.append(f"verified_stale>{pol.warn_if_verified_older_than_days}d")

        records.append(rec)

    return records


# -----------------------------
# Update mode
# -----------------------------
def _require_meta_marker_ack(write: bool, ack_marker: bool) -> None:
    """
    Guardrail: writing metadata/normalization without the marker convention will
    destroy the meaning of content freshness signals. Require an explicit ack.
    """
    if not write:
        return
    if ack_marker:
        return
    raise SystemExit(
        "Refusing to write without acknowledging metadata-commit convention.\n"
        "Re-run with --ack-meta-commit-marker and commit with:\n"
        f"  {SUGGESTED_TAGGED_COMMIT}\n"
    )


def update_files(
    repo_root: Path,
    cfg: ToolConfig,
    targets: list[str] | None,
    write: bool,
    set_content_date_from_git: bool,
    set_last_verified: date | None,
    set_verified_for: str | None,
    ack_meta_commit_marker: bool,
) -> list[FileRecord]:
    today = _iso_today()
    records = scan_files(repo_root, cfg, targets=targets)
    target_set = set(t.replace(os.sep, "/") for t in targets) if targets else None

    for rec in records:
        if rec.kind not in {"ipynb", "md"}:
            continue
        if rec.meta and rec.meta.ignore:
            continue
        if target_set is not None and rec.path not in target_set:
            continue

        meta = rec.meta or DLCMeta()

        # Build the desired metadata WITHOUT touching last_metadata_updated.
        if set_content_date_from_git and rec.last_content_updated is not None:
            meta.last_content_updated = rec.last_content_updated

        if set_last_verified is not None:
            meta.last_verified = set_last_verified
        if set_verified_for is not None:
            meta.verified_for = set_verified_for

        desired_base = meta_to_jsonable(meta)
        abs_path = repo_root / rec.path
        changed = False

        if rec.kind == "ipynb":
            nb, _raw, _has_dlc = read_ipynb_meta(abs_path)
            nb_meta = nb.setdefault("metadata", {})
            prev = nb_meta.get(DLC_NAMESPACE, {})
            if not isinstance(prev, dict):
                prev = {}

            merged_base = dict(prev)
            merged_base.update(desired_base)

            if merged_base != prev:
                changed = True
                if write:
                    _require_meta_marker_ack(write=True, ack_marker=ack_meta_commit_marker)

                    meta.last_metadata_updated = today
                    desired_final = meta_to_jsonable(meta)

                    merged_final = dict(prev)
                    merged_final.update(desired_final)
                    nb_meta[DLC_NAMESPACE] = merged_final
                    write_ipynb_meta(abs_path, nb)

        elif rec.kind == "md":
            text = abs_path.read_text(encoding="utf-8")
            fm, body, fm_error = read_md_frontmatter(text)
            if fm_error:
                msg = f"markdown_frontmatter_invalid: {fm_error}"
                if msg not in rec.errors:
                    rec.errors.append(msg)
                continue

            fm = fm or {}

            prev = fm.get(DLC_NAMESPACE, {})
            if not isinstance(prev, dict):
                prev = {}

            merged_base = dict(prev)
            merged_base.update(desired_base)

            if merged_base != prev:
                changed = True
                if write:
                    _require_meta_marker_ack(write=True, ack_marker=ack_meta_commit_marker)

                    meta.last_metadata_updated = today
                    desired_final = meta_to_jsonable(meta)

                    merged_final = dict(prev)
                    merged_final.update(desired_final)
                    fm[DLC_NAMESPACE] = merged_final
                    abs_path.write_text(dump_md_frontmatter(fm, body), encoding="utf-8")

        rec.would_change = changed
        rec.meta = meta
        rec.days_since_verified = compute_days_since(meta.last_verified, today)

    return records


# -----------------------------
# Notebook formatting
# -----------------------------
def normalize_notebooks(
    repo_root: Path,
    cfg: ToolConfig,
    targets: list[str] | None,
    write: bool,
    ack_meta_commit_marker: bool,
) -> list[FileRecord]:
    """
    Normalize notebooks deterministically (canonical nbformat JSON).
    This is intentionally separated from update() because it causes churn.
    """
    _require_meta_marker_ack(write=write, ack_marker=ack_meta_commit_marker)
    records = scan_files(repo_root, cfg, targets=targets)
    today = _iso_today()

    for rec in records:
        if rec.kind != "ipynb":
            continue
        if rec.meta and rec.meta.ignore:
            continue

        abs_path = repo_root / rec.path
        try:
            nb, _raw, _has_dlc = read_ipynb_meta(abs_path)
            nbformat.validate(nb)

            if not notebook_is_normalized(abs_path, nb):
                rec.would_change = True
                if write:
                    # Update embedded maintenance timestamp
                    meta = rec.meta or DLCMeta()
                    meta.last_metadata_updated = today

                    nb_meta = nb.setdefault("metadata", {})
                    prev = nb_meta.get(DLC_NAMESPACE, {})
                    if not isinstance(prev, dict):
                        prev = {}
                    merged = dict(prev)
                    merged.update(meta_to_jsonable(meta))
                    nb_meta[DLC_NAMESPACE] = merged

                    # Write to persist metadata update (still canonical)
                    write_ipynb_meta(abs_path, nb)
                    rec.meta = meta

        except Exception as e:
            rec.errors.append(f"normalize_failed: {e}")

    return records


# -----------------------------
# Output formatting
# -----------------------------


def summarize(records: list[FileRecord]) -> dict[str, int]:
    return {
        "files": len(records),
        "warnings": sum(1 for r in records if r.warnings),
        "errors": sum(1 for r in records if r.errors),
        "missing_metadata": sum(1 for r in records if "missing_metadata" in r.warnings),
        "missing_last_verified": sum(1 for r in records if "missing_last_verified" in r.warnings),
        "content_stale": sum(1 for r in records if any(w.startswith("content_stale") for w in r.warnings)),
        "verified_stale": sum(1 for r in records if any(w.startswith("verified_stale") for w in r.warnings)),
    }


def to_markdown(report: Report, cfg: ToolConfig) -> str:
    pol = cfg.policy
    t = report.totals
    lines: list[str] = []

    lines.append("# 🌡️ DeepLabCut freshness report\n")
    lines.append(f"Generated: {report.generated_at.isoformat()}\n")
    lines.append(f"Schema: v{report.schema_version}\n\n")

    lines.append("## Summary\n")
    lines.append(f"- Files scanned: **{t['files']}**\n")
    lines.append(f"- Files with warnings: **{t['warnings']}**\n")
    lines.append(f"- Files with scanning errors: **{t['errors']}**\n")
    lines.append(f"- Missing metadata: **{t['missing_metadata']}**\n")
    lines.append(f"- Missing last_verified: **{t['missing_last_verified']}**\n")
    lines.append(f"- Content-stale (> {pol.warn_if_content_older_than_days}d): **{t['content_stale']}**\n")
    lines.append(f"- Verification-stale (> {pol.warn_if_verified_older_than_days}d): **{t['verified_stale']}**\n\n")

    def fmt_date(d: date | None) -> str:
        return d.isoformat() if d else "-"

    warn_recs = [r for r in report.records if r.warnings and not (r.meta and r.meta.ignore)]
    warn_recs.sort(
        key=lambda r: (
            -(r.days_since_verified or -1),
            -(r.days_since_content_update or -1),
            r.path,
        )
    )

    if warn_recs:
        lines.append("## Warnings\n")
        for r in warn_recs:
            meta = r.meta
            lines.append(f"- **{r.path}** ({r.kind})\n")
            lines.append(
                f"  - last_content_updated: {fmt_date(r.last_content_updated)} "
                f"(days: {r.days_since_content_update if r.days_since_content_update is not None else '-'})\n"
            )
            if r.last_git_touched:
                lines.append(f"  - last_git_touched: {fmt_date(r.last_git_touched)}\n")
            if meta and meta.last_metadata_updated:
                lines.append(f"  - last_metadata_updated: {fmt_date(meta.last_metadata_updated)}\n")
            lv = meta.last_verified if meta else None
            lines.append(
                f"  - last_verified: {fmt_date(lv)} "
                f"(days: {r.days_since_verified if r.days_since_verified is not None else '-'})\n"
            )
            if meta and meta.verified_for:
                lines.append(f"  - verified_for: {meta.verified_for}\n")
            if meta and meta.tier:
                lines.append(f"  - tier: {meta.tier}\n")
            lines.append(f"  - warnings: {', '.join(r.warnings)}\n")
            if r.errors:
                lines.append(f"  - errors: {', '.join(r.errors)}\n")
        lines.append("\n")

    err_recs = [r for r in report.records if r.errors]
    if err_recs:
        lines.append("## Scan errors\n")
        for r in err_recs:
            lines.append(f"- **{r.path}**: {', '.join(r.errors)}\n")
        lines.append("\n")

    lines.append("## Notes\n")
    lines.append("- 'Out of date' does not necessarily mean 'broken'. Use this as a triage signal.\n")
    lines.append(
        "- last_git_touched / last_content_updated are computed from git history. "
        "last_verified is human-controlled.\n\n"
    )
    lines.append(
        "- In `check` mode, scan/parsing errors are reported for visibility but do not "
        "fail by default unless strict mode is enabled or they trigger an enforced policy rule.\n"
    )
    return "".join(lines)


def write_outputs(report: Report, cfg: ToolConfig, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{OUTPUT_FILENAME}.json"
    md_path = out_dir / f"{OUTPUT_FILENAME}.md"

    payload = report.model_dump(mode="json")

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report, cfg), encoding="utf-8")
    return json_path, md_path


# -----------------------------
# Check enforcement
# -----------------------------
def enforce(cfg: ToolConfig, records: list[FileRecord]) -> list[str]:
    pol = cfg.policy
    violations: list[str] = []
    today = _iso_today()

    for r in records:
        if r.meta and r.meta.ignore:
            continue
        if r.kind not in {"ipynb", "md"}:
            continue

        has_invalid_metadata = "invalid_metadata" in (r.warnings or [])

        if match_allowlist(r.path, pol.require_metadata):
            if has_invalid_metadata:
                violations.append(f"{r.path}: invalid metadata")
            elif r.meta is None:
                violations.append(f"{r.path}: missing metadata")

        if match_allowlist(r.path, pol.require_recent_verification):
            if has_invalid_metadata:
                violations.append(f"{r.path}: invalid metadata")
            else:
                lv = r.meta.last_verified if r.meta else None
                if lv is None:
                    violations.append(f"{r.path}: missing last_verified")
                else:
                    days = (today - lv).days
                    if days > pol.warn_if_verified_older_than_days:
                        violations.append(
                            f"{r.path}: last_verified is {days}d old (> {pol.warn_if_verified_older_than_days}d)"
                        )

        if r.kind == "ipynb" and match_allowlist(r.path, pol.require_notebook_normalized):
            if "notebook_not_normalized" in (r.warnings or []):
                violations.append(f"{r.path}: notebook is not normalized (run update/format)")

    return violations


# -----------------------------
# CLI
# -----------------------------


def parse_date_token(token: str) -> date:
    token = token.strip().lower()
    if token in {"today", "now"}:
        return _iso_today()
    return date.fromisoformat(token)


def collect_scan_issues(records: list[FileRecord], target: Literal["errors", "warnings"]) -> list[str]:
    items: list[str] = []
    for r in records:
        for e in getattr(r, target, []):
            items.append(f"{r.path}: {e}")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepLabCut checks tool (docs + notebooks)")
    parser.add_argument("--config", default=str(DEFAULT_CFG), help="Path to YAML config file")
    parser.add_argument(
        "--no-step-summary",
        action="store_true",
        help="Do not write to GITHUB_STEP_SUMMARY",
    )
    parser.add_argument("--out-dir", default=f"tmp/{OUTPUT_FILENAME}", help="Directory to write outputs")

    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="Generate staleness report (read-only)")
    rep.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of relative file paths to scan (limits scan to these files)",
    )

    chk = sub.add_parser(
        "check",
        help=(
            "Run scans + policy checks (read-only). "
            "Fails on enforced policy violations; scan errors are non-fatal by default."
        ),
    )
    chk.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of relative file paths to scan (limits scan to these files)",
    )
    chk.add_argument(
        "--strict-mode",
        action="store_true",
        help="Enable failure on scan/parsing errors (overrides config for this run)",
    )

    up = sub.add_parser("update", help="Update metadata/frontmatter (write mode requires --write)")
    up.add_argument(
        "--write",
        action="store_true",
        help="Actually write changes (otherwise dry-run)",
    )
    up.add_argument(
        "--set-content-date-from-git",
        action="store_true",
        help="Set embedded last_content_updated from computed git content date",
    )
    up.add_argument("--targets", nargs="*", help="Optional list of relative file paths to update")
    up.add_argument("--set-last-verified", default=None, help="YYYY-MM-DD or 'today'")
    up.add_argument("--set-verified-for", default=None, help="String like 3.0.0rc13")
    up.add_argument(
        "--ack-meta-commit-marker",
        action="store_true",
        help=f"Acknowledge that you will commit changes using marker: {META_COMMIT_MARKER}",
    )

    norm = sub.add_parser(
        "normalize",
        help="Normalize notebooks deterministically (write mode requires --write)",
    )
    norm.add_argument(
        "--write",
        action="store_true",
        help="Actually write changes (otherwise dry-run)",
    )
    norm.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of relative notebook paths to normalize",
    )
    norm.add_argument(
        "--ack-meta-commit-marker",
        action="store_true",
        help=f"Acknowledge that you will commit changes using marker: {META_COMMIT_MARKER}",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = Path(args.config)
    repo_root = find_repo_root(Path.cwd())
    cfg = load_config(config_path)
    out_dir = Path(args.out_dir)

    if args.cmd in {"report", "check"}:
        records = scan_files(repo_root, cfg, targets=getattr(args, "targets", None))
    elif args.cmd == "update":
        lv = parse_date_token(args.set_last_verified) if args.set_last_verified else None
        records = update_files(
            repo_root,
            cfg,
            targets=args.targets,
            write=bool(args.write),
            set_content_date_from_git=bool(args.set_content_date_from_git),
            set_last_verified=lv,
            set_verified_for=args.set_verified_for,
            ack_meta_commit_marker=bool(args.ack_meta_commit_marker),
        )
        if args.write:
            print(f"\nSuggested commit message:\n  {SUGGESTED_TAGGED_COMMIT}\n")

    else:  # normalize
        records = normalize_notebooks(
            repo_root,
            cfg,
            targets=args.targets,
            write=bool(args.write),
            ack_meta_commit_marker=bool(args.ack_meta_commit_marker),
        )
        if args.write:
            print(f"\nSuggested commit message:\n  {SUGGESTED_TAGGED_COMMIT}\n")

    report = Report(
        generated_at=datetime.now(timezone.utc),
        repo_root=str(repo_root),
        config_path=str(config_path),
        totals=summarize(records),
        records=records,
    )

    json_path, md_path = write_outputs(report, cfg, out_dir)

    # Emit GitHub Actions job summary if available
    emit_summary = not getattr(args, "no_step_summary", False)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if emit_summary and step_summary and md_path.exists():
        try:
            content = md_path.read_text(encoding="utf-8")
            # snippet = "\n".join(content.splitlines()[:220]) + "\n"
            snippet = "\n".join(content.splitlines()[:]) + "\n"
            Path(step_summary).write_text(snippet, encoding="utf-8")
        except Exception:
            pass

    scan_errors = collect_scan_issues(records, target="errors")
    if scan_errors:
        print("\nScan errors detected (non-fatal by default):")
        for item in scan_errors[:20]:
            print(f"- {item}")
        if len(scan_errors) > 20:
            print(f"... and {len(scan_errors) - 20} more (see report for full details)")

    if args.cmd == "check":
        violations = enforce(cfg, records)
        if violations:
            print("Policy violations:")
            for v in violations:
                print(f"- {v}")
            return 2
        strict_mode = bool((args.strict_mode) or cfg.policy.fail_on_scan_errors)
        if strict_mode and scan_errors:
            print("Strict mode enabled: failing due to scan/parsing errors.")
            return 1

    # Non-zero if metadata parsing errors occurred for non-report/check commands
    if args.cmd not in {"report", "check"} and any(r.errors for r in records):
        return 1
    else:
        print("\nReport generated:")
        print(f"- JSON: {json_path}")
        print(f"- Markdown: {md_path}")

        if any(r.warnings for r in records):
            print("Warnings detected; see report for details.")
        if any(r.errors for r in records):
            print("Scan errors detected; see report for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
