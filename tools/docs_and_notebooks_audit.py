"""
DeepLabCut docs audit export tool (validated / extensible).

Purpose
-------
Read audit metadata from the `deeplabcut` namespace in Markdown frontmatter and
notebook-level metadata, validate selected fields against enums/schema, and
export a CSV register with docs metadata and review notes to help drive documentation maintenance.

Supported metadata fields
-------------------------
From `deeplabcut:` this tool currently validates and exports:
- visibility
- status
- recommendation (with fallback alias: review_decision)
- last_verified (pass-through)
- notes (pass-through, but preserved from previous CSV if present even if updated in source)

Example metadata
----------------
Markdown frontmatter:

---
deeplabcut:
  visibility: online
  status: viable
  recommendation: keep
  last_verified: 2026-04-15
---

Notebook metadata:
{
  "metadata": {
    "deeplabcut": {
      "visibility": "online",
      "status": "viable",
      "recommendation": "keep",
      "last_verified": "2026-04-15"
    }
  }
}

Usage
-----
python tools/docs_audit_export.py \
  --config tools/docs_and_notebooks_report_config.yml \
  --out docs/_meta/docs_audit_register.csv

python tools/docs_audit_export.py \
  --targets docs/gui/ docs/recipes/*.md examples/COLAB/*.ipynb
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import nbformat
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------
DLC_NAMESPACE = "deeplabcut"
DEFAULT_CONFIG = Path("tools/docs_and_notebooks_report_config.yml")
DEFAULT_OUTPUT = Path("docs/_meta/docs_audit_register.csv")
GLOB_CHARS = set("*?[")
FRONTMATTER_RE = re.compile(r"^---\s*$")

# Generated columns owned by this tool. Any additional columns present in an
# existing CSV are preserved as human columns.
GENERATED_COLUMNS = [
    "path",
    "kind",
    "metadata_present",
    "visibility",
    "status",
    "recommendation",
    "last_verified",
    "parse_error",
    "validation_error",
    "notes",
]

DEFAULT_INCLUDE = [
    "docs/**/*.md",
    "docs/**/*.markdown",
    "docs/**/*.ipynb",
    "examples/**/*.ipynb",
    "README.md",
    "CONTRIBUTING.md",
]
DEFAULT_EXCLUDE = [
    ".git/**",
    ".github/**",
    "**/.ipynb_checkpoints/**",
    "**/node_modules/**",
    "**/.venv/**",
]

FileKind = Literal["md", "ipynb", "other"]
TargetKind = Literal["invalid", "file", "dir", "glob"]


# -----------------------------------------------------------------------------
# Enums / schema
# -----------------------------------------------------------------------------
class Visibility(str, Enum):
    """How discoverable the page is in the documentation surface."""

    ONLINE = "online"  # current docs surface / discoverable
    UNLISTED = "unlisted"  # intentionally available but not surfaced in nav
    ARCHIVED = "archived"  # only discoverable via archive/historical area
    ORPHANED = "orphaned"  # not listed and no supported inbound links


class Status(str, Enum):
    """Current health / lifecycle state of the page."""

    VIABLE = "viable"  # current and acceptable
    REVIEW_NEEDED = "review_needed"  # needs human review before decision
    OUTDATED = "outdated"  # content exists but is stale / drifted
    DEPRECATED = "deprecated"  # not preferred; replacement exists/coming
    ARCHIVED = "archived"  # kept for historical or niche reference
    REMOVED = "removed"  # removed from active docs surface


class Recommendation(str, Enum):
    """Recommended next action for the page."""

    KEEP = "keep"  # content is fine as-is; no action needed
    VERIFY = "verify"  # content and/or formatting could use verification
    UPDATE = "update"  # requires content update to be considered viable
    MOVE = "move"  # move to a more appropriate location
    MERGE = "merge"  # merge into another page
    ARCHIVE = "archive"  # if deprecated, archive before removal
    REMOVE = "remove"  # remove from repository (can be resurrected from git history if needed)


class AuditMetadata(BaseModel):
    """
    Strictly validate only the fields this exporter owns.

    Keep extra metadata allowed so the deeplabcut namespace can still contain
    other fields used by the main checks tool or future workflows.
    """

    model_config = ConfigDict(extra="allow")

    visibility: Visibility | None = None
    status: Status | None = None
    recommendation: Recommendation | None = None
    last_verified: str | None = None
    notes: str | None = None


class TargetSpec(TypedDict):
    raw: str
    normalized: str
    kind: TargetKind


class FieldSpec(BaseModel):
    """Describes how a CSV column maps from deeplabcut metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    column: str
    source_keys: list[str]
    extractor: Callable[[AuditMetadata, dict[str, Any]], str] | None = None


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(column="visibility", source_keys=["visibility"]),
    FieldSpec(column="status", source_keys=["status"]),
    FieldSpec(column="recommendation", source_keys=["recommendation", "review_decision"]),
    FieldSpec(column="last_verified", source_keys=["last_verified"]),
    FieldSpec(column="notes", source_keys=["notes"]),
]


# -----------------------------------------------------------------------------
# Path / target helpers
# -----------------------------------------------------------------------------
def normalize_target_spec(spec: str, repo_root: Path) -> str:
    s = spec.strip()
    if not s:
        return s
    s = s.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    p = Path(s)
    if p.is_absolute():
        try:
            s = str(p.resolve().relative_to(repo_root)).replace(os.sep, "/")
        except ValueError:
            s = str(p).replace(os.sep, "/")
    s = re.sub(r"/+", "/", s)
    if len(s) > 1:
        s = s.rstrip("/")
    return s


def compile_target_specs(targets: list[str] | None, repo_root: Path) -> list[TargetSpec] | None:
    if not targets:
        return None
    specs: list[TargetSpec] = []
    for raw in targets:
        normalized = normalize_target_spec(raw, repo_root)
        if not normalized:
            specs.append({"raw": raw, "normalized": "", "kind": "invalid"})
            continue
        if any(ch in normalized for ch in GLOB_CHARS):
            specs.append({"raw": raw, "normalized": normalized, "kind": "glob"})
            continue
        if raw.endswith(("/", "\\")):
            specs.append({"raw": raw, "normalized": normalized, "kind": "dir"})
            continue
        candidate = repo_root / normalized
        specs.append(
            {
                "raw": raw,
                "normalized": normalized,
                "kind": "dir" if candidate.exists() and candidate.is_dir() else "file",
            }
        )
    return specs


def target_spec_matches_path(rel_path: str, spec: TargetSpec) -> bool:
    rel_path = rel_path.replace("\\", "/")
    kind = spec["kind"]
    normalized = spec["normalized"]
    if kind == "invalid":
        return False
    if kind == "file":
        return rel_path == normalized
    if kind == "dir":
        return rel_path == normalized or rel_path.startswith(normalized + "/")
    if kind == "glob":
        return fnmatch.fnmatchcase(rel_path, normalized)
    return False


def target_matches(rel_path: str, specs: list[TargetSpec] | None) -> bool:
    return True if specs is None else any(target_spec_matches_path(rel_path, spec) for spec in specs)


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(start), capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        return Path(proc.stdout.strip()).resolve()
    raise RuntimeError("Could not locate repository root")


def file_kind(path: Path) -> FileKind:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "md"
    if suffix == ".ipynb":
        return "ipynb"
    return "other"


# -----------------------------------------------------------------------------
# Config / discovery
# -----------------------------------------------------------------------------
def load_scan_patterns(config_path: Path | None) -> tuple[list[str], list[str]]:
    if not config_path or not config_path.exists():
        return DEFAULT_INCLUDE, DEFAULT_EXCLUDE

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scan = raw.get("scan") or {}
    include = scan.get("include") or DEFAULT_INCLUDE
    exclude = scan.get("exclude") or DEFAULT_EXCLUDE
    return include, exclude


def is_excluded(rel_path: str, exclude_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in exclude_patterns)


def iter_candidate_paths(
    repo_root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    targets: list[str] | None = None,
) -> list[Path]:
    specs = compile_target_specs(targets, repo_root)
    matches: dict[str, Path] = {}
    for pattern in include_patterns:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            rel = str(path.resolve().relative_to(repo_root)).replace(os.sep, "/")
            if is_excluded(rel, exclude_patterns):
                continue
            if not target_matches(rel, specs):
                continue
            matches[rel] = path.resolve()
    return [matches[k] for k in sorted(matches)]


# -----------------------------------------------------------------------------
# Metadata readers (minimal duplication via namespace dispatch)
# -----------------------------------------------------------------------------
def read_md_frontmatter(text: str) -> tuple[dict | None, str | None]:
    lines = text.splitlines(keepends=True)
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return None, None

    end_idx = None
    for i in range(1, min(len(lines), 5000)):
        if FRONTMATTER_RE.match(lines[i]):
            end_idx = i
            break
    if end_idx is None:
        return None, "unterminated_markdown_frontmatter"

    fm_text = "".join(lines[1:end_idx])
    try:
        fm = yaml.safe_load(fm_text) if fm_text.strip() else {}
    except Exception as exc:
        return None, f"markdown_frontmatter_yaml_error: {exc}"
    if not isinstance(fm, dict):
        return None, "markdown_frontmatter_not_mapping"
    return fm, None


def read_container(path: Path) -> tuple[dict | None, str | None]:
    """
    Return the top-level metadata container for a file.
    - Markdown: YAML frontmatter mapping
    - Notebook: notebook.metadata mapping
    """
    kind = file_kind(path)

    if kind == "md":
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return None, f"read_failed: {exc}"
        return read_md_frontmatter(text)

    if kind == "ipynb":
        try:
            nb = nbformat.read(str(path), as_version=4)
            meta = getattr(nb, "metadata", {}) or {}
        except Exception as exc:
            return None, f"notebook_read_failed: {exc}"
        if not isinstance(meta, dict):
            return None, "notebook_metadata_not_mapping"
        return meta, None

    return None, None


def read_dlc_namespace(path: Path) -> tuple[dict | None, str | None]:
    container, error = read_container(path)
    if error:
        return None, error
    if container is None:
        return None, None
    raw = container.get(DLC_NAMESPACE)
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "deeplabcut_namespace_not_mapping"
    return raw, None


# -----------------------------------------------------------------------------
# Validation / normalization
# -----------------------------------------------------------------------------
def build_validation_input(raw_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Map raw deeplabcut metadata into the schema-owned keys.
    This is where aliases are resolved to canonical names.
    """
    payload: dict[str, Any] = {}
    for spec in FIELD_SPECS:
        for key in spec.source_keys:
            if key in raw_meta and raw_meta.get(key) not in {None, ""}:
                payload[spec.column] = raw_meta.get(key)
                break
    return payload


def validate_metadata(raw_meta: dict[str, Any] | None) -> tuple[AuditMetadata | None, str | None]:
    if raw_meta is None:
        return None, None
    try:
        validated = AuditMetadata.model_validate(build_validation_input(raw_meta))
        return validated, None
    except ValidationError as exc:
        messages = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return None, "; ".join(messages)


def extract_field_value(spec: FieldSpec, validated: AuditMetadata | None, raw_meta: dict[str, Any]) -> str:
    if spec.extractor is not None:
        return spec.extractor(validated, raw_meta)
    if validated is not None:
        value = getattr(validated, spec.column, None)
        if isinstance(value, Enum):
            return value.value
        return "" if value is None else str(value)
    # If validation failed, still emit raw/aliased value when present for easier triage.
    for key in spec.source_keys:
        if key in raw_meta and raw_meta.get(key) is not None:
            return str(raw_meta.get(key))
    return ""


# -----------------------------------------------------------------------------
# CSV merge / preserve human annotations
# -----------------------------------------------------------------------------
def load_existing_rows(csv_path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not csv_path.exists():
        return {}, []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = {row.get("path", ""): row for row in reader if row.get("path")}
        existing_columns = reader.fieldnames or []
    extra_columns = [c for c in existing_columns if c not in GENERATED_COLUMNS]
    return rows, extra_columns


def merged_row(base: dict[str, Any], previous: dict[str, str] | None, extra_columns: Iterable[str]) -> dict[str, Any]:
    row = dict(base)

    if previous:
        prev_notes = (previous.get("notes") or "").strip()
        scanned_notes = (row.get("notes") or "").strip()
        if prev_notes and scanned_notes and prev_notes != scanned_notes:
            print(f"WARNING: Notes conflict for {row['path']}:")
            print(f"-  Previous: {prev_notes}")
            print(f"-  Scanned:  {scanned_notes}")
            print("Preserving previous notes and ignoring scanned notes.")
        # Preserve human notes if present, otherwise keep scanned notes
        row["notes"] = prev_notes if prev_notes else scanned_notes

    for col in extra_columns:
        row[col] = previous.get(col, "") if previous else ""

    return row


# -----------------------------------------------------------------------------
# Row building / export
# -----------------------------------------------------------------------------
def build_row(repo_root: Path, path: Path) -> dict[str, Any]:
    rel = str(path.resolve().relative_to(repo_root)).replace(os.sep, "/")
    kind = file_kind(path)
    raw_meta, parse_error = read_dlc_namespace(path)
    raw_meta = raw_meta or {}
    metadata_present = bool(raw_meta)
    validated, validation_error = validate_metadata(raw_meta if metadata_present else None)

    row: dict[str, Any] = {
        "path": rel,
        "kind": kind,
        "metadata_present": "true" if metadata_present else "false",
        "parse_error": parse_error or "",
        "validation_error": validation_error or "",
        # "notes": "",
    }

    for spec in FIELD_SPECS:
        row[spec.column] = extract_field_value(spec, validated, raw_meta)
    return row


def export_csv(
    repo_root: Path, include: list[str], exclude: list[str], out_path: Path, targets: list[str] | None
) -> int:
    candidates = iter_candidate_paths(repo_root, include, exclude, targets=targets)
    existing_rows, extra_columns = load_existing_rows(out_path)

    rows = []
    for path in candidates:
        base = build_row(repo_root, path)
        previous = existing_rows.get(base["path"])
        rows.append(merged_row(base, previous, extra_columns))

    fieldnames = list(GENERATED_COLUMNS) + [c for c in extra_columns if c not in GENERATED_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} records to {out_path}")
    invalid = sum(1 for row in rows if row.get("validation_error"))
    parse_fail = sum(1 for row in rows if row.get("parse_error"))
    if invalid or parse_fail:
        print(f"Validation issues: {invalid}; parse issues: {parse_fail}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export DeepLabCut audit metadata to CSV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Optional path to scan config YAML")
    parser.add_argument("--root", default=".", help="Repository root or path inside the repository")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="CSV output path")
    parser.add_argument(
        "--targets",
        nargs="*",
        help=(
            "Optional repo-relative targets to limit the export. Supports exact files, "
            "directories, and glob patterns (e.g. docs/page.md, docs/gui/, 'docs/**/*.md')."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = find_repo_root(Path(args.root))
    config_path = Path(args.config)
    include, exclude = load_scan_patterns(config_path)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    return export_csv(repo_root, include, exclude, out_path, targets=args.targets)


if __name__ == "__main__":
    raise SystemExit(main())
