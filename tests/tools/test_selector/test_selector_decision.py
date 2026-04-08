# tests/tools/test_selector/test_selector_decision.py
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.test_selector_config import (
    CATEGORY_RULES,
    CategoryRule,
    prefix,
    validate_category_rules,
)


def assert_lanes(res, *, skip=False, docs=False, fast=False, full=False):
    assert res.lanes.skip is skip
    assert res.lanes.docs is docs
    assert res.lanes.fast is fast
    assert res.lanes.full is full


def test_fail_safe_on_empty_changes(selector):
    res = selector.decide([])

    assert_lanes(res, full=True)
    assert "no_changed_files_or_diff_unavailable" in res.reasons
    assert res.lane_reasons["full"] == ["no_changed_files_or_diff_unavailable"]


def test_docs_only(selector):
    files = ["docs/index.md", "docs/guide/intro.md", "_config.yml"]
    res = selector.decide(files)

    assert_lanes(res, docs=True)
    assert res.pytest_paths == []
    assert res.functional_scripts == []
    assert "category:docs" in res.reasons
    assert res.lane_reasons["docs"] == ["category:docs"]


def test_full_suite_trigger_pyproject_preserves_docs_lane(selector):
    files = ["pyproject.toml", "docs/index.md"]
    res = selector.decide(files)

    assert_lanes(res, full=True, docs=True)
    assert "full_suite_trigger" in res.reasons
    assert "category:docs" in res.reasons
    assert res.lane_reasons["full"] == [
        "full_suite_trigger",
        "full_suite_trigger_count:1",
    ]


def test_full_suite_trigger_tests_folder(selector):
    files = ["tests/test_something.py"]
    res = selector.decide(files)

    assert_lanes(res, full=True)
    assert "full_suite_trigger" in res.reasons
    assert res.lane_reasons["full"] == [
        "full_suite_trigger",
        "full_suite_trigger_count:1",
    ]


def test_fast_core(selector):
    files = ["deeplabcut/core/some_module.py"]
    res = selector.decide(files)

    assert_lanes(res, fast=True)

    # core rule should include these paths (subset check)
    assert "tests/core/" in res.pytest_paths
    assert "tests/utils/" in res.pytest_paths
    # assert res.functional_scripts == [] # not empty, but we don't need to specify exact scripts here

    assert "category:core" in res.reasons
    assert res.lane_reasons["fast"] == ["category:core"]

    # Provenance should attribute selected pytest roots to the core category.
    assert "tests/core/" in res.provenance.pytest
    assert res.provenance.pytest["tests/core/"] == ["core"]


def test_fast_multianimal_includes_functional(selector):
    files = ["deeplabcut/pose_estimation_pytorch/multianimal/foo.py"]
    res = selector.decide(files)

    assert_lanes(res, fast=True)

    assert "tests/test_predict_multianimal.py" in res.pytest_paths
    assert "examples/testscript_tensorflow_multi_animal.py" in res.functional_scripts

    assert "multianimal" in res.provenance.pytest["tests/test_predict_multianimal.py"]
    assert "multianimal" in res.provenance.scripts["examples/testscript_tensorflow_multi_animal.py"]


def test_fast_ci_workflows_uses_full_suite(selector):
    files = [".github/workflows/ci.yml"]
    res = selector.decide(files)

    assert_lanes(res, full=True)


def test_no_category_matched_is_full(selector):
    files = ["some/unknown/place/file.xyz"]
    res = selector.decide(files)

    assert_lanes(res, full=True)
    assert "no_category_matched" in res.reasons
    assert res.lane_reasons["full"] == ["no_category_matched"]


def test_docs_and_core_run_both_lanes(selector):
    files = ["docs/index.md", "deeplabcut/core/a.py"]
    res = selector.decide(files)

    assert_lanes(res, docs=True, fast=True)
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

    assert_lanes(res, fast=True)

    # No duplicates
    assert len(res.pytest_paths) == len(set(res.pytest_paths))
    assert len(res.functional_scripts) == len(set(res.functional_scripts))

    # Sorted
    assert res.pytest_paths == sorted(res.pytest_paths)
    assert res.functional_scripts == sorted(res.functional_scripts)


# ----------------------------
# Validation of category rules
# ----------------------------
def test_category_rule_rejects_empty_name():
    with pytest.raises(ValidationError, match="Rule name must not be empty"):
        CategoryRule(
            name="",
            match_any=[prefix("docs/")],
        )


def test_category_rule_rejects_invalid_name():
    with pytest.raises(ValidationError, match=r"Rule name must match"):
        CategoryRule(
            name="docs-rule",
            match_any=[prefix("docs/")],
        )


def test_category_rule_requires_non_empty_match_any():
    with pytest.raises(ValidationError, match="at least 1 item|at least one predicate"):
        CategoryRule(
            name="docs",
            match_any=[],
        )


def test_category_rule_rejects_non_callable_match_any():
    with pytest.raises(ValidationError, match="callable"):
        CategoryRule(
            name="docs",
            match_any=[123],  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("pytest_paths", "/absolute/path.py"),
        ("pytest_paths", "../escape.py"),
        ("functional_scripts", "/absolute/script.py"),
        ("functional_scripts", "../escape_script.py"),
    ],
)
def test_category_rule_rejects_invalid_repo_relative_paths(field_name, bad_value):
    kwargs = {
        "name": "docs",
        "match_any": [prefix("docs/")],
        "pytest_paths": [],
        "functional_scripts": [],
    }
    kwargs[field_name] = [bad_value]

    with pytest.raises(ValidationError, match="repo-relative|path traversal|absolute path"):
        CategoryRule(**kwargs)


def test_validate_category_rules_rejects_duplicate_names():
    rules = [
        CategoryRule(name="docs", match_any=[prefix("docs/")]),
        CategoryRule(name="docs", match_any=[prefix("more-docs/")]),
    ]

    with pytest.raises(ValueError, match="Duplicate CategoryRule name"):
        validate_category_rules(rules)


def test_lint_only_changes_select_skip_lane(selector):
    files = [".pre-commit-config.yaml"]
    res = selector.decide(files)

    assert_lanes(res, skip=True)
    assert res.pytest_paths == []
    assert res.functional_scripts == []

    assert "lint_only" in res.reasons
    assert "skip" in res.lane_reasons
    assert "lint_only" in res.lane_reasons["skip"]


def test_validate_selected_paths_escalates_to_full_on_missing(selector, tmp_path: Path):
    # Build a minimal repo dir with none of the selected paths present.
    repo = tmp_path / "repo"
    repo.mkdir()

    res = selector.SelectorResult(
        lanes=selector.LaneSelection(fast=True),
        pytest_paths=["tests/does_not_exist.py"],
        functional_scripts=["examples/missing_script.py"],
        provenance=selector.SelectionProvenance(
            pytest={"tests/does_not_exist.py": ["core"]},
            scripts={"examples/missing_script.py": ["core"]},
        ),
        reasons=["category:core"],
        changed_files=["deeplabcut/core/foo.py"],
        lane_reasons={"fast": ["category:core"]},
    )

    out = selector.validate_selected_paths(res, repo)

    assert out.lanes.fast is False
    assert out.lanes.full is True

    assert out.pytest_paths == []
    assert out.functional_scripts == []
    assert out.provenance.pytest == {}
    assert out.provenance.scripts == {}

    assert "missing_selected_paths" in out.reasons
    assert any(r.startswith("pytest:tests/does_not_exist.py") for r in out.reasons)
    assert any(r.startswith("script:examples/missing_script.py") for r in out.reasons)

    assert "full" in out.lane_reasons


def test_validate_selected_paths_keeps_fast_when_paths_exist(selector, tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "examples").mkdir(parents=True)

    (repo / "tests" / "test_ok.py").write_text("def test_ok(): pass\n")
    (repo / "examples" / "script_ok.py").write_text("print('ok')\n")

    res = selector.SelectorResult(
        lanes=selector.LaneSelection(fast=True),
        pytest_paths=["tests/test_ok.py"],
        functional_scripts=["examples/script_ok.py"],
        provenance=selector.SelectionProvenance(
            pytest={"tests/test_ok.py": ["core"]},
            scripts={"examples/script_ok.py": ["core"]},
        ),
        reasons=["category:core"],
        changed_files=["deeplabcut/core/foo.py"],
        lane_reasons={"fast": ["category:core"]},
    )

    out = selector.validate_selected_paths(res, repo)

    assert out.lanes.fast is True
    assert out.lanes.full is False
    assert out.pytest_paths == ["tests/test_ok.py"]
    assert out.functional_scripts == ["examples/script_ok.py"]


# --------------------------------------
# Current config validity & sanity checks
# --------------------------------------


def test_current_category_rules_are_typed_models():
    assert CATEGORY_RULES
    assert all(isinstance(rule, CategoryRule) for rule in CATEGORY_RULES)


def test_current_category_rules_pass_cross_rule_validation():
    validate_category_rules(CATEGORY_RULES)


def test_current_category_rule_names_are_unique():
    names = [rule.name for rule in CATEGORY_RULES]
    assert len(names) == len(set(names))


def test_current_category_rules_have_matchers():
    assert all(rule.match_any for rule in CATEGORY_RULES)


def test_required_category_rules_exist():
    names = {rule.name for rule in CATEGORY_RULES}
    assert "docs" in names
    assert "core" in names


def test_docs_rule_exists_once():
    docs_rules = [rule for rule in CATEGORY_RULES if rule.name == "docs"]
    assert len(docs_rules) == 1


def test_current_selected_paths_exist():
    repo_root = Path(__file__).resolve().parents[3]
    missing = []

    for rule in CATEGORY_RULES:
        for path in rule.pytest_paths:
            if not (repo_root / path).exists():
                missing.append((rule.name, "pytest", path))
        for path in rule.functional_scripts:
            if not (repo_root / path).exists():
                missing.append((rule.name, "script", path))

    assert missing == []
