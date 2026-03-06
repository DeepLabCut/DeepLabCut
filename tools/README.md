# Developer tools useful for maintaining the repository

This document summarizes the developer tooling and workflows used in this repo.

> **Quick start (recommended)**
>
> ```bash
> python -m pip install -U pip
> pip install -U pytest coverage pre-commit
> ```
>
> If you prefer faster dependency installs, you can use `uv` (optional):
>
> ```bash
> # Install uv (see https://docs.astral.sh/uv/ for platform-specific installers)
> uv --version
> ```

---

## 1) Pre-commit (recommended)

Enable the repository hooks locally:

```bash
pre-commit install
```

Run on all files:

```bash
pre-commit run --all-files
```

---

## 2) License headers

Code headers can be standardized by running:

```bash
python tools/update_license_headers.py
```

Run from the repository root. Update `NOTICE.yml` to change header content.

---

## 3) Running tests locally

### Run the full test suite

```bash
pytest
```

### Run a specific test module or folder

```bash
pytest tests/test_auxiliaryfunctions.py
pytest tests/core/
```

### Coverage

```bash
coverage run -m pytest
coverage report
```

---

## 4) Intelligent test selection (local + CI)

The repository includes a deterministic test selection tool to reduce CI runtime by running only relevant tests based on changed files.

### What it outputs

The selector produces a single, unambiguous **plan** plus structured lists:

- `plan`: `docs_only` | `fast` | `full`
- `pytest_paths`: list of pytest path args (JSON)
- `functional_scripts`: list of scripts to run (JSON)

It also emits audit metadata:

- `reasons`: why the decision was taken
- `changed_files`: files considered for the decision

### Run locally (no CI env required)

Print decision as JSON:

```bash
python tools/test_selector.py --json
```

Write a report (JSON + Markdown) under `tmp/test-selection/`:

```bash
python tools/test_selector.py --report-dir tmp/test-selection --json
```

> In GitHub Actions, the workflow typically adds `--write-github-output` and `--write-summary`.

### Notes

- The selector is **fail-safe**: if changes cannot be determined or are ambiguous, it selects `plan=full`.
- Rules are intentionally location-based and centralized in the selector script:
  - Conservative “full-suite triggers”
  - Small category mapping rules

---

## 5) Docs: Jupyter Book build (local)

The repo uses Jupyter Book for docs:

```bash
python -m pip install -U pip
python -m pip install .[docs]
jupyter-book build .
```

`.github/workflows/build-book.yml` is the canonical CI implementation.

---

s## 6) Troubleshooting tips

- If a workflow run is unexpectedly selecting `full`, check the selector report:
  - `tmp/test-selection/decision.md`
  - `tmp/test-selection/selection.json`
- If targeted tests fail due to missing dependencies, either:
  - broaden the fast-lane install (e.g., install extras), or
  - adjust selection rules to only include tests that run in the minimal environment.
