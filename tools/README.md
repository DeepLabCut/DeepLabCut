# Developer tools useful for maintaining the repository

As developer you'll need:

```bash
pip install coverage pytest fnmatch black
```

## Code headers

The code headers can be standardized by running

``` bash
python tools/update_license_headers.py
```

from the repository root.

You can edit the `NOTICE.yml` to update the header.


## Workflow for contributing/checking your code

```bash
black .
```

## Running the tests (locally)

We use the pytest framework. You can just run:

```bash
pytest
```

For coverage run:

```
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
