"""DeepLabCut docs & notebooks automated checks tool.

Goals
-----
- SAFE by default: read-only operations in CI (report/check).
- Idempotent updates (update mode) that only touch:
    * Notebook-level metadata for .ipynb (never cells/outputs)
    * YAML frontmatter for .md docs (optional)
- Uses pydantic schemas with explicit schema_version for validation.

Terminology
-----------
last_git_updated
    Computed from git history (last commit touching the file).

last_verified
    Human-controlled date indicating the file was verified to work/be accurate.

verified_for
    Human-controlled string, typically the project version (e.g. 3.0.0rc13).

tier
    Optional classification (left unset by default; do not auto-populate).

Usage modes
-----------
Report (read-only):
    python .github/tools/docs_and_notebooks_check.py report

Check (read-only; may fail based on config allowlists):
    python .github/tools/docs_and_notebooks_check.py check

Update git-updated fields (write mode; requires --write):
    python .github/tools/docs_and_notebooks_check.py update --write --only-git-date

Update verification fields for selected targets (write mode):
    python .github/tools/docs_and_notebooks_check.py update --write --targets docs/page.md \
        --set-last-verified today --set-verified-for 3.0.0rc13

Configuration
-------------
Uses .github/tools/docs_and_notebooks_report_config.yml by default.  

Outputs  
-------  
- docs_nb_checks.json: machine-readable report  
- docs_nb_checks.md: human-readable summary

Notes for CI
------------
- Ensure actions/checkout uses fetch-depth: 0 (or sufficiently deep),
  otherwise git log may not see history.
- Requires:
  - pydantic
  - PyYAML 
  - nbformat
  to be installed in the environment. 
  Recommended : install in CI job directly (pip install pydantic pyyaml nbformat) rather than adding to requirements, since these are only needed for this tool.
"""
# .github/tools/docs_and_notebooks_check.py
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

try:
    from pydantic import BaseModel, Field, ValidationError, ConfigDict
except Exception:  # pragma: no cover
    raise RuntimeError("Pydantic is required to run this script")
try:
    import nbformat
    from nbformat.validator import NotebookValidationError
except Exception:
    raise RuntimeError("nbformat is required to read/write .ipynb files")

SCHEMA_VERSION = 1
DLC_NAMESPACE = "deeplabcut"
OUTPUT_FILENAME = "docs_nb_checks"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = (SCRIPT_DIR / "docs_and_notebooks_report_config.yml")
# -----------------------------
# Pydantic schemas
# -----------------------------

class DLCMeta(BaseModel):
    """Metadata embedded in files under the `deeplabcut` namespace."""
    model_config = ConfigDict(extra="allow")

    last_git_updated: Optional[date] = None
    last_verified: Optional[date] = None
    verified_for: Optional[str] = None
    tier: Optional[str] = None
    ignore: bool = False
    notes: Optional[str] = None


class ScanConfig(BaseModel):
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    warn_if_git_older_than_days: int = 365
    warn_if_verified_older_than_days: int = 365
    missing_last_verified_is_warning: bool = True

    # Allowlists for strict checks (start empty; ratchet later)
    require_metadata: List[str] = Field(default_factory=list)
    require_recent_verification: List[str] = Field(default_factory=list)
    
    require_notebook_normalized: List[str] = Field(default_factory=list)


class ToolConfig(BaseModel):
    version: int = 1
    scan: ScanConfig
    policy: PolicyConfig


class FileRecord(BaseModel):
    path: str
    kind: str  # ipynb | md | other

    # Computed from git
    last_git_updated: Optional[date] = None

    # Read from file metadata/frontmatter
    meta: Optional[DLCMeta] = None

    # Derived
    days_since_git_update: Optional[int] = None
    days_since_verified: Optional[int] = None

    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # If update mode would change file
    would_change: bool = False


class Report(BaseModel):
    schema_version: int = SCHEMA_VERSION
    generated_at: datetime
    repo_root: str
    config_path: str

    totals: Dict[str, int]
    records: List[FileRecord]


# -----------------------------
# Helpers
# -----------------------------

def _iso_today() -> date:
    return datetime.now(timezone.utc).date()


def _run_git(args: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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


def glob_paths(repo_root: Path, patterns: List[str]) -> List[Path]:
    results: List[Path] = []
    for pat in patterns:
        results.extend(repo_root.glob(pat))
    return sorted({p.resolve() for p in results if p.is_file()})


def is_excluded(rel_path: str, exclude_patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_patterns)


def file_kind(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".ipynb":
        return "ipynb"
    if s in {".md", ".markdown"}:
        return "md"
    return "other"


def git_last_updated(repo_root: Path, rel_path: str) -> Optional[date]:
    code, out, _err = _run_git(["log", "-1", "--format=%cI", "--", rel_path], cwd=repo_root)
    if code != 0 or not out:
        return None
    try:
        return datetime.fromisoformat(out).date()
    except Exception:
        return None


FRONTMATTER_RE = re.compile(r"^---\s*$")


def read_md_frontmatter(text: str) -> Tuple[Optional[dict], str]:
    lines = text.splitlines(keepends=True)
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return None, text

    end_idx = None
    for i in range(1, min(len(lines), 5000)):
        if FRONTMATTER_RE.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return None, text

    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])

    if yaml is None:
        raise RuntimeError("PyYAML is required to parse Markdown frontmatter")

    fm = yaml.safe_load(fm_text) if fm_text.strip() else {}
    if not isinstance(fm, dict):
        return None, text
    return fm, body


def dump_md_frontmatter(frontmatter: dict, body: str) -> str:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write Markdown frontmatter")
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    return "---\n" + fm_text + "---\n" + body.lstrip("\n")


def read_ipynb_meta(path: Path) -> tuple[Any, dict]:
    """
    Read a notebook using nbformat. Returns (notebook_node, deeplabcut_meta_dict).
    """
    nb = nbformat.read(str(path), as_version=4)

    meta = getattr(nb, "metadata", {}) or {}
    dlc_meta = meta.get(DLC_NAMESPACE, {})
    if not isinstance(dlc_meta, dict):
        dlc_meta = {}
    return nb, dlc_meta

def notebook_is_normalized(path: Path, nb: Any) -> bool:
    original = path.read_text(encoding="utf-8")
    normalized = nbformat.writes(nb, version=4, indent=2, ensure_ascii=False) + "\n"
    return original == normalized

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

def parse_dlc_meta(raw: Any) -> Optional[DLCMeta]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        try:
            return DLCMeta.model_validate(raw)  # pydantic v2
        except AttributeError:
            return DLCMeta.parse_obj(raw)       # pydantic v1
        except ValidationError:
            return None
    return None


def meta_to_jsonable(meta: DLCMeta) -> dict:
    """
    Return JSON-serializable metadata (dates become ISO strings).
    This prevents json.dumps() from failing when writing .ipynb files.
    """
    try:
        # Pydantic v2: mode='json' converts date/datetime into ISO strings
        return meta.model_dump(mode="json", exclude_none=True)
    except AttributeError:
        # Pydantic v1: meta.json() encodes dates; parse back into dict
        return json.loads(meta.json(exclude_none=True))


def compute_days_since(d: Optional[date], today: date) -> Optional[int]:
    return None if d is None else (today - d).days


def match_allowlist(rel_path: str, allowlist: List[str]) -> bool:
    # Support exact matches or glob patterns
    return any(pat == rel_path or fnmatch.fnmatch(rel_path, pat) for pat in allowlist)


# -----------------------------
# Core scanning
# -----------------------------

def load_config(config_path: Path) -> ToolConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        return ToolConfig.model_validate(raw)  # pydantic v2
    except AttributeError:
        return ToolConfig.parse_obj(raw)       # pydantic v1


def scan_files(repo_root: Path, cfg: ToolConfig, targets: Optional[List[str]] = None) -> List[FileRecord]:
    today = _iso_today()
    paths = glob_paths(repo_root, cfg.scan.include)
    records: List[FileRecord] = []
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

        rec.last_git_updated = git_last_updated(repo_root, rel)
        rec.days_since_git_update = compute_days_since(rec.last_git_updated, today)

        try:
            if kind == "ipynb":
                nb, raw_meta = read_ipynb_meta(p)
                try:
                    nbformat.validate(nb)
                except NotebookValidationError as e:
                    rec.errors.append(f"nbformat_invalid: {e}")
                
                try:
                    if not notebook_is_normalized(p, nb):
                        rec.warnings.append("notebook_not_normalized")
                except Exception as e:
                    # Don't crash scan if a file has encoding/IO oddities
                    rec.errors.append(f"notebook_normalization_check_failed: {e}")

                rec.meta = parse_dlc_meta(raw_meta)
            elif kind == "md":
                text = p.read_text(encoding="utf-8")
                fm, _body = read_md_frontmatter(text)
                raw = (fm or {}).get(DLC_NAMESPACE)
                rec.meta = parse_dlc_meta(raw)
            else:
                rec.meta = None
        except Exception as e:
            rec.errors.append(f"metadata_read_failed: {e}")

        if rec.meta and rec.meta.ignore:
            records.append(rec)
            continue

        last_verified = rec.meta.last_verified if rec.meta else None
        rec.days_since_verified = compute_days_since(last_verified, today)

        pol = cfg.policy

        if rec.days_since_git_update is not None and rec.days_since_git_update > pol.warn_if_git_older_than_days:
            rec.warnings.append(f"git_stale>{pol.warn_if_git_older_than_days}d")

        if last_verified is None and pol.missing_last_verified_is_warning:
            rec.warnings.append("missing_last_verified")
        elif rec.days_since_verified is not None and rec.days_since_verified > pol.warn_if_verified_older_than_days:
            rec.warnings.append(f"verified_stale>{pol.warn_if_verified_older_than_days}d")

        if kind in {"ipynb", "md"} and rec.meta is None:
            rec.warnings.append("missing_metadata")

        records.append(rec)

    return records


# -----------------------------
# Update mode
# -----------------------------

def update_files(
    repo_root: Path,
    cfg: ToolConfig,
    targets: Optional[List[str]],
    write: bool,
    only_git_date: bool,
    set_last_verified: Optional[date],
    set_verified_for: Optional[str],
) -> List[FileRecord]:
    today = _iso_today()
    records = scan_files(repo_root, cfg)
    target_set = set(t.replace(os.sep, "/") for t in targets) if targets else None

    for rec in records:
        if rec.kind not in {"ipynb", "md"}:
            continue
        if rec.meta and rec.meta.ignore:
            continue
        if target_set is not None and rec.path not in target_set:
            continue

        meta = rec.meta or DLCMeta()

        # Always update last_git_updated to computed value (if available)
        if rec.last_git_updated is not None:
            meta.last_git_updated = rec.last_git_updated

        if not only_git_date:
            if set_last_verified is not None:
                meta.last_verified = set_last_verified
            if set_verified_for is not None:
                meta.verified_for = set_verified_for

        desired = meta_to_jsonable(meta)
        abs_path = repo_root / rec.path
        changed = False

        if rec.kind == "ipynb":
            nb, _raw = read_ipynb_meta(abs_path)
            nb_meta = nb.setdefault("metadata", {})
            prev = nb_meta.get(DLC_NAMESPACE, {})
            if not isinstance(prev, dict):
                prev = {}
            merged = dict(prev)
            merged.update(desired)
            if merged != prev:
                nb_meta[DLC_NAMESPACE] = merged
                changed = True
                if write:
                    write_ipynb_meta(abs_path, nb)

        elif rec.kind == "md":
            text = abs_path.read_text(encoding="utf-8")
            fm, body = read_md_frontmatter(text)
            fm = fm or {}
            prev = fm.get(DLC_NAMESPACE, {})
            if not isinstance(prev, dict):
                prev = {}
            merged = dict(prev)
            merged.update(desired)
            if merged != prev:
                fm[DLC_NAMESPACE] = merged
                changed = True
                if write:
                    abs_path.write_text(dump_md_frontmatter(fm, body), encoding="utf-8")

        rec.would_change = changed
        rec.meta = meta
        rec.days_since_verified = compute_days_since(meta.last_verified, today)

    return records


# -----------------------------
# Output formatting
# -----------------------------

def summarize(records: List[FileRecord]) -> Dict[str, int]:
    return {
        "files": len(records),
        "warnings": sum(1 for r in records if r.warnings),
        "errors": sum(1 for r in records if r.errors),
        "missing_metadata": sum(1 for r in records if "missing_metadata" in r.warnings),
        "missing_last_verified": sum(1 for r in records if "missing_last_verified" in r.warnings),
        "git_stale": sum(1 for r in records if any(w.startswith("git_stale") for w in r.warnings)),
        "verified_stale": sum(1 for r in records if any(w.startswith("verified_stale") for w in r.warnings)),
    }


def to_markdown(report: Report, cfg: ToolConfig) -> str:
    pol = cfg.policy
    t = report.totals
    lines: List[str] = []

    lines.append("# DeepLabCut staleness report\n")
    lines.append(f"Generated: {report.generated_at.isoformat()}\n")
    lines.append(f"Schema: v{report.schema_version}\n\n")

    lines.append("## Summary\n")
    lines.append(f"- Files scanned: **{t['files']}**\n")
    lines.append(f"- Files with warnings: **{t['warnings']}**\n")
    lines.append(f"- Files with errors: **{t['errors']}**\n")
    lines.append(f"- Missing metadata: **{t['missing_metadata']}**\n")
    lines.append(f"- Missing last_verified: **{t['missing_last_verified']}**\n")
    lines.append(f"- Git-stale (> {pol.warn_if_git_older_than_days}d): **{t['git_stale']}**\n")
    lines.append(f"- Verification-stale (> {pol.warn_if_verified_older_than_days}d): **{t['verified_stale']}**\n\n")

    def fmt_date(d: Optional[date]) -> str:
        return d.isoformat() if d else "-"

    warn_recs = [r for r in report.records if r.warnings and not (r.meta and r.meta.ignore)]
    warn_recs.sort(key=lambda r: (-(r.days_since_verified or -1), -(r.days_since_git_update or -1), r.path))

    if warn_recs:
        lines.append("## Warnings\n")
        for r in warn_recs:
            meta = r.meta
            lines.append(f"- **{r.path}** ({r.kind})\n")
            lines.append(f"  - last_git_updated: {fmt_date(r.last_git_updated)} "
                        f"(days: {r.days_since_git_update if r.days_since_git_update is not None else '-'})\n")
            lv = meta.last_verified if meta else None
            lines.append(f"  - last_verified: {fmt_date(lv)} "
                        f"(days: {r.days_since_verified if r.days_since_verified is not None else '-'})\n")
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
        lines.append("## Errors\n")
        for r in err_recs:
            lines.append(f"- **{r.path}**: {', '.join(r.errors)}\n")
        lines.append("\n")

    lines.append("## Notes\n")
    lines.append("- 'Out of date' does not necessarily mean 'broken'. Use this as a triage signal.\n")
    lines.append("- last_git_updated is computed from git history. last_verified is human-controlled.\n\n")
    return "".join(lines)


def write_outputs(report: Report, cfg: ToolConfig, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{OUTPUT_FILENAME}.json"
    md_path = out_dir / f"{OUTPUT_FILENAME}.md"

    try:
        payload = report.model_dump(mode="json")  # pydantic v2
    except AttributeError:
        payload = json.loads(report.json())       # pydantic v1

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report, cfg), encoding="utf-8")
    return json_path, md_path


# -----------------------------
# Check enforcement
# -----------------------------

def enforce(cfg: ToolConfig, records: List[FileRecord]) -> List[str]:
    pol = cfg.policy
    violations: List[str] = []
    today = _iso_today()

    for r in records:
        if r.meta and r.meta.ignore:
            continue
        if r.kind not in {"ipynb", "md"}:
            continue

        if match_allowlist(r.path, pol.require_metadata) and r.meta is None:
            violations.append(f"{r.path}: missing metadata")

        if match_allowlist(r.path, pol.require_recent_verification):
            lv = r.meta.last_verified if r.meta else None
            if lv is None:
                violations.append(f"{r.path}: missing last_verified")
            else:
                days = (today - lv).days
                if days > pol.warn_if_verified_older_than_days:
                    violations.append(f"{r.path}: last_verified is {days}d old "
                                      f"(> {pol.warn_if_verified_older_than_days}d)")

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DeepLabCut checks tool (docs + notebooks)")
    parser.add_argument("--config", default=str(DEFAULT_CFG), help="Path to YAML config file")
    parser.add_argument("--out-dir", default=f"tmp/{OUTPUT_FILENAME}", help="Directory to write outputs")

    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="Generate staleness report (read-only)")
    rep.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of relative file paths to scan (limits scan to these files)",
    )

    chk = sub.add_parser("check", help="Run policy checks (read-only; may exit non-zero)")
    chk.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of relative file paths to scan (limits scan to these files)",
    )

    up = sub.add_parser("update", help="Update metadata/frontmatter (write mode requires --write)")
    up.add_argument("--write", action="store_true", help="Actually write changes (otherwise dry-run)")
    up.add_argument("--only-git-date", action="store_true", help="Only update last_git_updated")
    up.add_argument("--targets", nargs="*", help="Optional list of relative file paths to update")
    up.add_argument("--set-last-verified", default=None, help="YYYY-MM-DD or 'today'")
    up.add_argument("--set-verified-for", default=None, help="String like 3.0.0rc13")

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = Path(args.config)
    repo_root = find_repo_root(Path.cwd())
    cfg = load_config(config_path)
    out_dir = Path(args.out_dir)

    if args.cmd in {"report", "check"}:
        records = scan_files(repo_root, cfg, targets=getattr(args, "targets", None))
    else:
        lv = parse_date_token(args.set_last_verified) if args.set_last_verified else None
        records = update_files(
            repo_root,
            cfg,
            targets=args.targets,
            write=bool(args.write),
            only_git_date=bool(args.only_git_date),
            set_last_verified=lv,
            set_verified_for=args.set_verified_for,
        )

    report = Report(
        generated_at=datetime.now(timezone.utc),
        repo_root=str(repo_root),
        config_path=str(config_path),
        totals=summarize(records),
        records=records,
    )

    json_path, md_path = write_outputs(report, cfg, out_dir)

    # Emit GitHub Actions job summary if available
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and md_path.exists():
        try:
            content = md_path.read_text(encoding="utf-8")
            snippet = "\n".join(content.splitlines()[:220]) + "\n"
            Path(step_summary).write_text(snippet, encoding="utf-8")
        except Exception:
            pass

    if args.cmd == "check":
        violations = enforce(cfg, records)
        if violations:
            print("Policy violations:")
            for v in violations:
                print(f"- {v}")
            return 2

    # Non-zero if metadata parsing errors occurred (except in 'report' mode)  
    if args.cmd != "report" and any(r.errors for r in records):  
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())