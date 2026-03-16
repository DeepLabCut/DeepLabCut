# Docs & Notebooks Checks Tool

This tool scans DeepLabCut documentation pages and notebooks and produces **two independent signals**:

- **`last_content_updated`**: computed from git history as the last *meaningful content* update **excluding metadata-only commits**.
- **`last_verified`**: a human-controlled date indicating the content was verified to work/be accurate.

In addition, the tool can optionally track:

- **`last_metadata_updated`**: when the tool last performed a metadata/normalization write (helps explain “file changed” without implying content changed).
- **`verified_for`**: a human-controlled string indicating what the content was verified against (e.g. `3.0.0rc13`).

The tool is designed to be:

- **Safe by default**: CI should run **read-only** modes (`report` / `check`).
- **Deterministic**: stable outputs and normalized notebook formatting when explicitly requested.
- **Future-proof**: versioned Pydantic schemas (`schema_version`).

---

## What gets scanned

Default include patterns are defined in `tools/docs_and_notebooks_report_config.yml`.
Typical patterns include:

- `examples/COLAB/**/*.ipynb`
- `examples/JUPYTER/**/*.ipynb`
- `docs/**/*.md`
- `docs/**/*.ipynb` (if notebooks are added under docs)

You can further restrict the scan via `--targets`.

---

## Metadata storage locations

### Notebooks (`.ipynb`)

The tool **only** reads/writes **top-level notebook metadata** under the `deeplabcut` namespace.

> [!IMPORTANT]
> It never edits notebook cells, outputs, or execution counts.

Example (excerpt):

```json
{
  "metadata": {
    "deeplabcut": {
      "last_content_updated": "2020-01-01",
      "last_metadata_updated": "2026-03-05",
      "last_verified": "2026-02-20",
      "verified_for": "3.0.0rc13",
      "ignore": false
    }
  }
}
```

> [!NOTE]
> `tier` is intentionally optional and is not auto-populated.

### Markdown (`.md`)

The tool reads/writes YAML frontmatter at the top of the file (if present):

```yaml
---
deeplabcut:
  last_content_updated: 2020-01-01
  last_metadata_updated: 2026-03-05
  last_verified: 2026-02-20
  verified_for: 3.0.0rc13
  ignore: false
---
```

If a doc page has **no** frontmatter, the tool can still report staleness (read-only), and `update` can add/modify metadata when explicitly requested.

---

## The metadata-commit marker (critical)

Because metadata updates and notebook normalization can rewrite files, they would normally make git (correctly) report that the file was “updated now”.

To preserve a meaningful **`last_content_updated`**, **all metadata-only / normalization commits must include the marker**:

- **Marker**: `META_COMMIT_MARKER` (see `tools/docs_and_notebooks_check.py`)
- **Suggested commit message**: `SUGGESTED_META_COMMIT_MESSAGE`

When you run `update --write` or `normalize --write`, the tool will:

- Require `--ack-meta-commit-marker` (guardrail)
- Print a suggested commit message

> [!WARNING]
> If the marker changes in the future, previous iterations still HAVE to be acknowledged to avoid false positives.

---

## Commands

### 1) Report (read-only)

Generate a report (does not modify files):

```bash
python tools/docs_and_notebooks_check.py report
```

Writes (by default):

- `tmp/docs_nb_checks/docs_nb_checks.json`
- `tmp/docs_nb_checks/docs_nb_checks.md`

### 2) Check (read-only; may fail)

Run policy checks. By default, CI will not fail unless allowlists are configured.

```bash
python tools/docs_and_notebooks_check.py check
```

The allowlists live in `tools/docs_and_notebooks_report_config.yml`.
They are currently empty, but can help enforce stricter policies once populated (start empty; "ratchet" later).

### 3) Update metadata (write mode; explicit intent)

> [!WARNING]
> `update --write` modifies tracked files. Intended for maintainers (manual), not CI.

#### 3a) Set `last_content_updated` from git (excluding meta commits)

```bash
python tools/docs_and_notebooks_check.py update   --write   --set-content-date-from-git   --ack-meta-commit-marker
```

#### 3b) Set verification fields (human-controlled)

```bash
python tools/docs_and_notebooks_check.py update   --write   --targets docs/page.md examples/JUPYTER/foo.ipynb   --set-last-verified today   --set-verified-for 3.0.0rc13   --ack-meta-commit-marker
```

> Tip: omit `--targets` to operate on all scanned files.

### 4) Normalize notebooks (explicit churn)

> [!IMPORTANT]
> Notebook normalization rewrites the notebook JSON into a canonical form.
> As such, it is provided as a separate command.

Dry-run (shows which files *would* change):

```bash
python tools/docs_and_notebooks_check.py normalize --targets docs/notebook.ipynb
```

Write:

```bash
python tools/docs_and_notebooks_check.py normalize   --write   --targets docs/notebook.ipynb   --ack-meta-commit-marker
```

---

## CI integration

Recommended CI usage:

- Run `report` on PRs and upload the outputs as artifacts.
- Run `check` once allowlists are populated (start empty to avoid failures).

> [!IMPORTANT]
> Use `actions/checkout` with `fetch-depth: 0` (or sufficiently deep) so `git log` sees history; shallow clones can cause missing or fallback timestamps.

Dependencies required for this tool (install in the CI job):

```bash
pip install pydantic pyyaml nbformat
```

---

## Troubleshooting

- If you see `content_date_fallback_to_git_touched`, it usually means one of:
  - The checkout history is too shallow, or
  - *All* commits touching the file are metadata commits with the marker.

- If Pydantic raises `class-not-fully-defined` errors, ensure the tool calls `.model_rebuild()` for its models (this is already done in the tool).
