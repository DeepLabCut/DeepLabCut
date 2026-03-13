from __future__ import annotations

import pytest


def test_fail_safe_on_empty_changes(selector):
    res = selector.decide([])

    assert res.lanes.full is True
    assert res.lanes.fast is False
    assert res.lanes.docs is False
    assert res.lanes.skip is False
    assert "no_changed_files_or_diff_unavailable" in res.reasons
    assert res.lane_reasons["full"] == ["no_changed_files_or_diff_unavailable"]


def test_docs_only(selector):
    files = ["docs/index.md", "docs/guide/intro.md", "_config.yml"]
    res = selector.decide(files)

    assert res.lanes.docs is True
    assert res.lanes.fast is False
    assert res.lanes.full is False
    assert res.lanes.skip is False
    assert res.pytest_paths == []
    assert res.functional_scripts == []
    assert "category:docs" in res.reasons
    assert res.lane_reasons["docs"] == ["category:docs"]


def test_full_suite_trigger_pyproject_preserves_docs_lane(selector):
    files = ["pyproject.toml", "docs/index.md"]
    res = selector.decide(files)

    assert res.lanes.full is True
    assert res.lanes.docs is True
    assert res.lanes.fast is False
    assert res.lanes.skip is False
    assert "full_suite_trigger" in res.reasons
    assert "category:docs" in res.reasons
    assert res.lane_reasons["full"] == [
        "full_suite_trigger",
        "full_suite_trigger_count:1",
    ]


def test_full_suite_trigger_tests_folder(selector):
    files = ["tests/test_something.py"]
    res = selector.decide(files)

    assert res.lanes.full is True
    assert res.lanes.docs is False
    assert res.lanes.fast is False
    assert res.lanes.skip is False
    assert "full_suite_trigger" in res.reasons
    assert res.lane_reasons["full"] == [
        "full_suite_trigger",
        "full_suite_trigger_count:1",
    ]


def test_fast_core(selector):
    files = ["deeplabcut/core/some_module.py"]
    res = selector.decide(files)

    assert res.lanes.fast is True
    assert res.lanes.full is False
    assert res.lanes.docs is False
    assert res.lanes.skip is False

    # core rule should include these paths (subset check)
    assert "tests/core/" in res.pytest_paths
    assert "tests/utils/" in res.pytest_paths
    assert res.functional_scripts == []

    assert "category:core" in res.reasons
    assert res.lane_reasons["fast"] == ["category:core"]

    # Provenance should attribute selected pytest roots to the core category.
    assert "tests/core/" in res.provenance.pytest
    assert res.provenance.pytest["tests/core/"] == ["core"]


def test_fast_multianimal_includes_functional(selector):
    files = ["deeplabcut/pose_estimation_pytorch/multianimal/foo.py"]
    res = selector.decide(files)

    assert res.lanes.fast is True
    assert res.lanes.full is False
    assert res.lanes.docs is False
    assert res.lanes.skip is False

    assert "tests/test_predict_multianimal.py" in res.pytest_paths
    assert "examples/testscript_multianimal.py" in res.functional_scripts

    assert res.provenance.pytest["tests/test_predict_multianimal.py"] == ["multianimal"]
    assert res.provenance.scripts["examples/testscript_multianimal.py"] == [
        "multianimal"
    ]


def test_fast_ci_workflows_uses_minimal_pytest(selector):
    files = [".github/workflows/ci.yml"]
    res = selector.decide(files)

    assert res.lanes.fast is True
    assert res.lanes.full is False
    assert res.lanes.docs is False
    assert res.lanes.skip is False

    # ci_workflows explicitly maps to the minimal pytest path
    assert selector.MINIMAL_PYTEST[0] in res.pytest_paths
    assert res.lane_reasons["fast"] == ["category:ci_workflows"]
    assert res.provenance.pytest[selector.MINIMAL_PYTEST[0]] == ["ci_workflows"]


def test_no_category_matched_is_full(selector):
    files = ["some/unknown/place/file.xyz"]
    res = selector.decide(files)

    assert res.lanes.full is True
    assert res.lanes.fast is False
    assert res.lanes.docs is False
    assert res.lanes.skip is False
    assert "no_category_matched" in res.reasons
    assert res.lane_reasons["full"] == ["no_category_matched"]


def test_docs_and_core_run_both_lanes(selector):
    files = ["docs/index.md", "deeplabcut/core/a.py"]
    res = selector.decide(files)

    assert res.lanes.docs is True
    assert res.lanes.fast is True
    assert res.lanes.full is False
    assert res.lanes.skip is False

    assert "category:docs" in res.reasons
    assert "category:core" in res.reasons

    assert res.lane_reasons["docs"] == ["category:docs"]
    assert res.lane_reasons["fast"] == ["category:core"]

    assert "tests/core/" in res.pytest_paths
    assert "tests/utils/" in res.pytest_paths


def test_dedup_and_sorted_outputs(selector):
    # Force overlap: core includes tests/test_auxiliaryfunctions.py and
    # ci_tools contributes tests/tools/. Outputs should stay deduped and sorted.
    files = [
        "deeplabcut/core/a.py",
        "tools/whatever.py",
    ]
    res = selector.decide(files)

    assert res.lanes.fast is True
    assert res.lanes.full is False
    assert res.lanes.skip is False

    # No duplicates
    assert len(res.pytest_paths) == len(set(res.pytest_paths))
    assert len(res.functional_scripts) == len(set(res.functional_scripts))

    # Sorted
    assert res.pytest_paths == sorted(res.pytest_paths)
    assert res.functional_scripts == sorted(res.functional_scripts)
