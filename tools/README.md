# Developer tools useful for maintaining the repository

This document summarizes the developer tooling and workflows used in this repo.

```bash
pip install -e . --group dev
```

---

## 1) Pre-commit (recommended)

Enable the repository hooks locally:

```bash
pre-commit install
```

Run on all files:

Steering committee members may edit the `NOTICE.yml` to update the header.

## 2) Ruff cleanup helpers

For **local Ruff backlog work** (not a substitute for CI or pre-commit), see [Ruff cleanup helpers](ruff_cleanup_helpers.md). It documents `generate_ruff_report.py` (Markdown report from Ruff JSON) and `fix_e501_with_autopep8.py` (targeted long-line cleanup plus Ruff fix/format).

---

## 3) License headers

Code headers can be standardized by running:

Please follow the instructions in `CONTRIBUTING.md` for contributing to the codebase, including running tests and pre-commit checks before opening a pull request.

Run from the repository root. Update `NOTICE.yml` to change header content.

---

## 4) Running tests locally

### Run the full test suite

```bash
pytest
```

### Run a specific test module or folder

```bash
coverage run -m pytest
coverage report
```

## 5) Intelligent test selection (local + CI)

The repository includes a deterministic test-selection tool to reduce CI runtime by running only the relevant workflows and tests based on changed files.

### What it outputs

The selector emits **orthogonal workflow lanes** plus structured selections:

- `lanes`: which workflow lanes should run
  - `skip`: skip test execution entirely (for lint-only changes)
  - `docs`: run docs checks
  - `fast`: run targeted pytest paths and optional functional scripts
  - `full`: delegate to the full test workflow / matrix
- `pytest_paths`: list of pytest path arguments (JSON)
- `functional_scripts`: list of Python scripts to run (JSON)
- `provenance`: mapping from each selected test/script to the category rule(s) that selected it

It also emits audit metadata:

- `selected_workflows`: ordered list of enabled lanes (`skip`, `docs`, `fast`, `full`)
- `lane_reasons`: reasons for each enabled lane
- `diff_mode`: how the diff range was determined
- `reasons`: aggregate machine-readable reasons for the decision
- `changed_files`: files considered for the decision
- `schema_version`: output schema version

### Rule configuration

Routing rules are defined in `tools/test_selector_config.py`.

That file contains:

- reusable path predicate helpers such as `prefix(...)`, `suffix(...)`, `equals(...)`, `case_insensitive_match(...)`, and `all_of(...)`
- conservative `FULL_SUITE_TRIGGERS`
- `LINT_ONLY_FILES`
- validated `CATEGORY_RULES` built from the `CategoryRule` schema
- `CATEGORY_RULE_BY_NAME` for stable lookup of named rules such as `docs`

The current refactor keeps the rule predicates simple and location-based while validating the rule structure at import time.

### Run locally (no CI env required)

> [!IMPORTANT]
> Requires `pydantic>=2,<3`

Print the decision as JSON:

```bash
python tools/test_selector.py --json
```

Write the decision report (`selection.json` and `decision.md`) under `tmp/test-selection/`:

```bash
python tools/test_selector.py --report-dir tmp/test-selection --json
```

Write a GitHub job-summary-compatible Markdown report when `GITHUB_STEP_SUMMARY` is available:

```bash
python tools/test_selector.py --report-dir tmp/test-selection --write-summary
```

Override the diff range manually:

```bash
python tools/test_selector.py --base-sha <base> --head-sha <head> --json
```

In GitHub Actions, the workflow typically adds `--write-github-output` and `--write-summary`.

### Diff modes

The selector records how the diff was determined in `diff_mode`:

- `pr`: pull request diff using `merge-base(base, head)..head`
- `push`: push diff using `before..after`
- `manual`: explicit `--base-sha` / `--head-sha`
- `fallback`: fallback to `HEAD^..HEAD`
- `initial`: initial commit (`empty-tree..HEAD`)
- `fallback_no_head`: could not resolve `HEAD`

### Report files

The selector always writes report artifacts for transparency:

- `tmp/test-selection/selection.json`: machine-readable output
- `tmp/test-selection/decision.md`: human-readable summary with workflow lanes, reasons, explained changed files, selected tests, and provenance

These reports are especially useful when a change unexpectedly routes to `full`.

### Notes

- The selector can enable more than one lane at once. For example, a PR can legitimately enable both `docs` and `fast`, or `docs` and `full`.
- Docs changes are **orthogonal** to test routing: docs changes can enable the docs lane while still contributing selected tests/scripts if such rules are configured.
- `LINT_ONLY_FILES` are ignored for routing. If *only* lint-only files changed, the selector enables the `skip` lane.
- If category rules match changed files but do not contribute explicit tests/scripts, the selector can fall back to the minimal pytest set defined by `MINIMAL_PYTEST`.

### Troubleshooting the selector

If a workflow run is unexpectedly selecting `full`, check:

- `tmp/test-selection/decision.md`
- `tmp/test-selection/selection.json`
- `lane_reasons`
- `diff_mode`
- `changed_files`

Common causes include:

- a file matched a conservative full-suite trigger
- no category rule matched the routed files
- selected paths configured by a rule no longer exist in the repository
- diff resolution fell back because CI checkout history was incomplete

---

## 6) Docs: Jupyter Book build (local)

The repo uses Jupyter Book for docs:

```bash
python -m pip install -U pip
python -m pip install .[docs]
jupyter-book build .
```

`.github/workflows/build-book.yml` is the canonical CI implementation.

---

## 7) Testing the test selector

The selector has dedicated tests covering:

- decision behavior for docs / fast / full / skip routing
- provenance and deduplicated selections
- `CategoryRule` schema validation
- integrity checks for the currently defined rules

Run the selector-focused tests with:

```bash
pytest tests/tools/test_selector/
```

---

## 8) Troubleshooting tips

- If a workflow run is unexpectedly selecting `full`, inspect the selector reports first.
- If targeted tests fail due to missing dependencies, either:
  - broaden the fast-lane install (for example by installing required extras), or
  - adjust selection rules so that the fast lane only selects tests that run in the minimal environment.
- If manual diff selection is used, always pass both `--base-sha` and `--head-sha` together.
- In CI, ensure checkout history is deep enough for `merge-base` / `diff` operations (`fetch-depth: 0` is typically safest).
