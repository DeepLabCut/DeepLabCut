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

# Ruff Cleanup Helpers

This document describes two small developer-focused utilities that help contributors work through Ruff lint issues in a large Python codebase:

- `generate_ruff_report.py` — generate a readable Markdown report from Ruff JSON output
- `fix_e501_with_autopep8.py` — aggressively reduce `E501` (line-too-long) violations, then normalize with Ruff

These tools are intended for **local cleanup workflows**, **incremental lint adoption**, and **one-off contributor maintenance work**.
They are especially useful when a repository already has a non-trivial Ruff backlog and you want to:

1. understand what remains,
2. prioritize manual fixes,
3. and automate the highest-volume style cleanups safely enough for review.

---

## Who should use these tools?

These scripts are aimed at:

- contributors doing lint cleanup PRs,
- maintainers reducing legacy Ruff debt,
- developers triaging a large number of remaining violations,
- anyone who wants a more readable workflow than raw CLI output.

They are **not** intended to replace normal Ruff usage in CI or pre-commit. Instead, think of them as **cleanup helpers** around Ruff.

---

## What each script does

### `generate_ruff_report.py`

Runs Ruff in JSON mode and turns the results into a **human-readable Markdown report**.

It groups issues:

- by Ruff rule,
- then by file,
- then by line/column/message.

It also includes:

- a summary table,
- short hints for common rules,
- a suggested triage order,
- simple `code -g file:line` commands to jump into affected files.

This is useful when raw `ruff check` output is too noisy or when you want something that can be attached to an issue / PR / cleanup plan.
It also provides quick navigation and file open commands to help you jump into the right places in the codebase.

---

### `fix_e501_with_autopep8.py`

Finds files that still contain Ruff `E501` violations, then runs a narrow cleanup pipeline on those files only:

1. `autopep8` to aggressively reflow long lines,
2. `ruff check --fix --unsafe-fixes` to apply available lint fixes,
3. `ruff format` to normalize formatting.

This script is intentionally scoped to **files that Ruff already reports as having `E501`** so that it avoids unnecessary churn in unrelated files.

> [!WARNING]
> This can be rather aggressive. One known issue is for f-strings that are wrapped across multiple lines, which may produce a broken pattern such as:
> ```python
> f"some string with a {
>     var
> }"
> ```
> If that happens, search for lines where `}"` appears by itself with only indentation around it.
>
> A useful regex is:
> ```regex
> ^[ \t]*\}"[ \t]*$
> ```

---

## Requirements

### Required tools

Both scripts assume the following tools are available on your system PATH:

- `python`
- `ruff`

Additionally:

- `fix_e501_with_autopep8.py` also requires `autopep8`

### Install example

```bash
python -m pip install ruff autopep8
```

If you use `uv`:

```bash
uv add --dev ruff autopep8
```

---

## Script 1: `generate_ruff_report.py`

### Purpose

Generate a readable Markdown report from Ruff's JSON output.

### Typical usage

Run on the whole repository:

```bash
python generate_ruff_report.py . --output tmp/ruff-report.md
```

Run on selected paths only:

```bash
python generate_ruff_report.py src tests --output tmp/ruff-report.md
```

### Output

By default the script writes to:

```text
tmp/ruff-report.md
```

The output contains:

- total issue count,
- summary table by rule,
- short notes for common rules,
- suggested triage order,
- per-rule sections,
- per-file counts,
- detailed line/column/message tables,
- quick-open commands for VS Code.

### Example workflow

```bash
ruff check .
python generate_ruff_report.py . --output tmp/ruff-report.md
```

Open the Markdown report, pick a rule family (for example `F403`, `F405`, `F821`, `E722`, `B904`), and work through the files systematically.

---

## Script 2: `fix_e501_with_autopep8.py`

### Purpose

Reduce Ruff `E501` violations (`line-too-long`) using `autopep8`, then normalize those files with Ruff.

### Typical usage

Run on the whole repository:

```bash
python fix_e501_with_autopep8.py . --line-length 88
```

Run on selected paths only:

```bash
python fix_e501_with_autopep8.py src tests --line-length 100
```

Dry-run mode (show affected files only):

```bash
python fix_e501_with_autopep8.py . --line-length 88 --check
```

### What it does internally

For the given paths, the script:

1. runs Ruff in JSON mode,
2. extracts the set of files that still contain `E501`,
3. runs `autopep8` only on those files,
4. runs `ruff check --fix --unsafe-fixes` on the same files,
5. runs `ruff format` on those same files,
6. prints how many files still contain `E501` afterwards.

### Why this script is narrow by design

`E501` cleanup can create a lot of diff noise if you run formatters indiscriminately. This tool tries to keep the blast radius smaller by only touching files already flagged by Ruff for line length issues.

### Known caveat: malformed multiline f-strings

In some cases, aggressive line wrapping may produce a broken multiline f-string pattern such as:

```python
f"some string with a {
    var
}"
```

If that happens, search for lines where `}"` appears by itself with only indentation around it.

A useful regex is:

```regex
^[ \t]*\}"[ \t]*$
```

This can help you quickly find and manually repair those cases.

### Good use cases

- reducing a large backlog of `E501` violations before a more careful cleanup pass,
- "massaging" legacy code that was never formatter-cleaned consistently.

---

## Recommended workflow for contributors

If you are working on lint cleanup, a practical workflow is:

### 1. Generate a report

```bash
python generate_ruff_report.py . --output tmp/ruff-report.md
```

### 2. Reduce long lines first (optional but often useful)

```bash
python fix_e501_with_autopep8.py . --line-length 88
```

### 3. Re-run the report

```bash
python generate_ruff_report.py . --output tmp/ruff-report.md
```

### 4. Triage remaining issues manually

---

## Limitations

### `generate_ruff_report.py`

- only reports what Ruff emits,
- does not fix anything,
- hints are heuristic and intentionally brief.

### `fix_e501_with_autopep8.py`

- targets only `E501` files,
- depends on `autopep8` behavior,
- may create formatting diffs that require manual review,
- cannot infer semantic intent for every line wrap,
- may occasionally produce awkward formatting or broken multiline f-strings.

---

## Safety notes

Before committing results from `fix_e501_with_autopep8.py`:

1. run Ruff again,
2. run the relevant test suite,
3. scan diff hunks involving long strings / f-strings / messages,
4. review any surprising changes in error messages, docstrings, or string interpolation.

Suggested commands:

```bash
ruff check .
ruff format --check .
pytest
```

---

## Examples

### Generate a repo-wide manual-fix report

```bash
python generate_ruff_report.py . --output tmp/ruff-report.md
```

### Generate a report only for Python package code

```bash
python generate_ruff_report.py deeplabcut tests --output tmp/ruff-report.md
```

### See which files still have `E501`

```bash
python fix_e501_with_autopep8.py . --line-length 120 --check
```

### Reduce long-line issues, then review the remaining backlog

```bash
python fix_e501_with_autopep8.py . --line-length 120
python generate_ruff_report.py . --output tmp/ruff-report.md
```

---

## Summary

These scripts are small but practical helpers for maintaining a large Ruff-enabled Python repository:

- `generate_ruff_report.py` turns Ruff output into a human-readable action plan
- `fix_e501_with_autopep8.py` helps shrink `E501` noise before manual cleanup

Use them as **developer tools**, not as a substitute for understanding or reviewing changes.

---

## 4) Intelligent test selection (local + CI)

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

## 5) Docs: Jupyter Book build (local)

The repo uses Jupyter Book for docs:

```bash
python -m pip install -U pip
python -m pip install .[docs]
jupyter-book build .
```

`.github/workflows/build-book.yml` is the canonical CI implementation.

---

## 6) Testing the test selector

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

## 7) Troubleshooting tips

- If a workflow run is unexpectedly selecting `full`, inspect the selector reports first.
- If targeted tests fail due to missing dependencies, either:
  - broaden the fast-lane install (for example by installing required extras), or
  - adjust selection rules so that the fast lane only selects tests that run in the minimal environment.
- If manual diff selection is used, always pass both `--base-sha` and `--head-sha` together.
- In CI, ensure checkout history is deep enough for `merge-base` / `diff` operations (`fetch-depth: 0` is typically safest).
