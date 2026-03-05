# Docs & Notebooks Checks Tool

This tool scans DeepLabCut notebooks and documentation pages and reports two independent signals:

- **last_git_updated**: last commit date touching a file (computed from git history)
- **last_verified**: a human-controlled date indicating the content was verified to work/be accurate

It is designed to be **safe by default** (read-only in CI), and **future-proof** via versioned
pydantic schemas.

## Files scanned

Default patterns (see `.github/tools/docs_and_notebooks_report_config.yml`):

- `examples/COLAB/**/*.ipynb`
- `examples/JUPYTER/**/*.ipynb`
- `docs/**/*.md`
- `docs/**/*.ipynb` (if notebooks get added to docs; Jupyter Book supports this)

## Metadata locations

### Notebooks (`.ipynb`)

The tool only touches the notebook **top-level JSON metadata** under the `deeplabcut` namespace.

> [!IMPORTANT]
> It never edits cells, outputs, or execution counts.

Example (excerpt):

```json
{
  "metadata": {
    "deeplabcut": {
      "last_git_updated": "2026-03-05",
      "last_verified": "2026-02-20",
      "verified_for": "3.0.0rc13",
      "ignore": false
    }
  }
}
```


> [!NOTE]
> **Tier** is intentionally optional and is not auto-populated.

### Markdown files (`.md`)

The tool reads/writes YAML frontmatter at the top of the file:

```
---
deeplabcut:
  last_git_updated: 2026-03-05
  last_verified: 2026-02-20
  verified_for: 3.0.0rc13
  ignore: false
---
```
If a doc page has no frontmatter, the tool can still report potential staleness (read-only).

## Usage

### Report (read-only)

```
python .github/tools/docs_and_notebooks_check.py report
```

Writes (by default):  

- `tmp/docs_nb_checks/docs_nb_checks.json`  
- `tmp/docs_nb_checks/docs_nb_checks.md`

### Check (read-only, may fail)

Check only fails based on allowlists in tools/staleness_config.yml:

```
python .github/tools/docs_and_notebooks_check.py check
```

By default these are empty, so CI will not fail.

### Update (write mode)

> [!WARNING]
> This mode updates files in-place and should be used with caution. It is intended to be run manually by maintainers, not in CI.


Update only last_git_updated for all scanned files:
`python .github/tools/docs_and_notebooks_check.py update --write --only-git-date`

Set verification metadata for specific target files:

```
python .github/tools/docs_and_notebooks_check.py update --write --targets examples/JUPYTER/foo.ipynb \
  --set-last-verified today --set-verified-for 3.0.0rc13
```

### CI integration

Add a CI step that runs:
`python .github/tools/docs_and_notebooks_check.py report` 
and uploads the outputs as artifacts.
Optionally run check once allowlists are populated.

Important: Ensure actions/checkout uses a non-shallow clone (fetch-depth: 0) so git log
can compute last_git_updated reliably.