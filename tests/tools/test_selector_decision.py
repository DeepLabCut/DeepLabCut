from __future__ import annotations

import pytest


def test_fail_safe_on_empty_changes(selector):
    res = selector.decide([])
    assert res.plan == selector.Plan.FULL
    assert "no_changed_files_or_diff_unavailable" in res.reasons


def test_docs_only(selector):
    files = ["docs/index.md", "docs/guide/intro.md", "_config.yml"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.DOCS_ONLY
    assert res.pytest_paths == []
    assert res.functional_scripts == []
    assert "docs_only" in res.reasons


def test_full_suite_trigger_pyproject(selector):
    files = ["pyproject.toml", "docs/index.md"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FULL
    assert "full_suite_trigger" in res.reasons


def test_full_suite_trigger_tests_folder(selector):
    files = ["tests/test_something.py"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FULL
    assert "full_suite_trigger" in res.reasons


def test_fast_core(selector):
    files = ["deeplabcut/core/some_module.py"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FAST
    # core rule should include these paths (subset check)
    assert "tests/core/" in res.pytest_paths
    assert "tests/utils/" in res.pytest_paths
    assert res.functional_scripts == []
    assert any(r.startswith("category:") for r in res.reasons)


def test_fast_multianimal_includes_functional(selector):
    files = ["deeplabcut/pose_estimation_pytorch/multianimal/foo.py"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FAST
    assert "tests/test_predict_multianimal.py" in res.pytest_paths
    assert "examples/testscript_multianimal.py" in res.functional_scripts


def test_fast_ci_tools(selector):
    files = [".github/workflows/ci.yml"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FAST
    # should fall back to minimal pytest path via ci_tools rule
    assert selector.MINIMAL_PYTEST[0] in res.pytest_paths


def test_no_category_matched_is_full(selector):
    files = ["some/unknown/place/file.xyz"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FULL
    assert "no_category_matched" in res.reasons


def test_too_many_categories_is_full(selector):
    # docs + core + tools => 3 categories => FULL
    files = ["docs/index.md", "deeplabcut/core/a.py", "tools/something.py"]
    res = selector.decide(files)
    assert res.plan == selector.Plan.FULL
    assert any(r.startswith("too_many_categories:") for r in res.reasons)


def test_dedup_and_sorted_outputs(selector):
    # Force overlap: core includes tests/test_auxiliaryfunctions.py and MINIMAL_PYTEST also has it
    files = [
        "deeplabcut/core/a.py",
        "tools/whatever.py",
    ]  # core + ci_tools => 2 categories -> FAST
    res = selector.decide(files)
    assert res.plan == selector.Plan.FAST
    # No duplicates
    assert len(res.pytest_paths) == len(set(res.pytest_paths))
    # Sorted
    assert res.pytest_paths == sorted(res.pytest_paths)
    assert res.functional_scripts == sorted(res.functional_scripts)
