"""Test selector configuration."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

PathPred = Callable[[str], bool]


def prefix(*values: str) -> PathPred:
    """Match if path starts with any of the given prefixes."""
    prefixes = tuple(values)
    return lambda p: p.startswith(prefixes)


def suffix(*values: str) -> PathPred:
    """Match if path ends with any of the given suffixes."""
    suffixes = tuple(values)
    return lambda p: p.endswith(suffixes)


def equals(*values: str) -> PathPred:
    """Match if path equals any of the given exact values."""
    allowed = frozenset(values)
    return lambda p: p in allowed


def case_insensitive_match(*values: str) -> PathPred:
    """Case-insensitive substring match against any of the given values."""
    needles = tuple(v.lower() for v in values)
    return lambda p: any(n in p.lower() for n in needles)


def all_of(*preds: PathPred) -> PathPred:
    """Logical AND over predicates."""
    return lambda p: all(pred(p) for pred in preds)


# -----------------------------
#  Rules validation models
# -----------------------------
_RULE_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _validate_relpath_string(value: str, field_name: str) -> str:
    """Validate a repo-relative path string used in config."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} entries must be strings")

    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} entries must not be empty")

    value = value.replace("\\", "/")

    if value.startswith("/"):
        raise ValueError(f"{field_name} must be repo-relative, got absolute path: {value!r}")

    if re.match(r"^[A-Za-z]:/", value):
        raise ValueError(f"{field_name} must not be a Windows absolute path: {value!r}")

    parts = PurePosixPath(value).parts
    if ".." in parts:
        raise ValueError(f"{field_name} must not contain path traversal: {value!r}")

    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes: {value!r}")

    return value


class CategoryRule(BaseModel):
    """Validated test-selection category rule."""

    model_config = ConfigDict(extra="forbid")

    name: str
    match_any: list[PathPred] = Field(
        min_length=1,
        description=("List of predicates; if any predicate matches any changed file, the rule is triggered."),
    )
    pytest_paths: list[str] = Field(
        default_factory=list,
        description="Pytest paths selected if the rule is triggered.",
    )
    functional_scripts: list[str] = Field(
        default_factory=list,
        description="Functional test scripts selected if the rule is triggered.",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Rule name must not be empty")
        if not _RULE_NAME_RE.match(value):
            raise ValueError(f"Rule name must match ^[a-z0-9_]+$ (got {value!r})")
        return value

    @field_validator("match_any")
    @classmethod
    def validate_match_any(cls, preds: list[PathPred]) -> list[PathPred]:
        if not preds:
            raise ValueError("match_any must contain at least one predicate")
        for i, pred in enumerate(preds):
            if not callable(pred):
                raise TypeError(f"match_any[{i}] must be callable, got {type(pred).__name__}")
        return preds

    @field_validator("pytest_paths")
    @classmethod
    def validate_pytest_paths(cls, values: list[str]) -> list[str]:
        return [_validate_relpath_string(v, "pytest_paths") for v in values]

    @field_validator("functional_scripts")
    @classmethod
    def validate_functional_scripts(cls, values: list[str]) -> list[str]:
        return [_validate_relpath_string(v, "functional_scripts") for v in values]


def validate_category_rules(rules: list[CategoryRule]) -> list[CategoryRule]:
    """Validate cross-rule invariants."""
    seen: dict[str, int] = {}

    for idx, rule in enumerate(rules):
        if rule.name in seen:
            first_idx = seen[rule.name]
            raise ValueError(f"Duplicate CategoryRule name {rule.name!r} at indexes {first_idx} and {idx}")
        seen[rule.name] = idx

    return rules


# -----------------------------
# Configuration
# -----------------------------
MINIMAL_PYTEST = ["tests/test_auxiliaryfunctions.py"]

POSE_TF = "deeplabcut/pose_estimation_tensorflow/"
POSE_PT = "deeplabcut/pose_estimation_pytorch/"


# Conservative full-suite triggers: if any changed file matches, plan=FULL.
FULL_SUITE_TRIGGERS = [
    ("Tests files changed", prefix("tests/")),
    ("pyproject.toml changed", equals("pyproject.toml")),
    ("lockfile changed", suffix(".lock")),
    ("DEEPLABCUT.yaml changed", suffix("DEEPLABCUT.yaml")),
]


# Files that should be enforced by dedicated lint workflows, not by test selection
LINT_ONLY_FILES = {
    ".pre-commit-config.yaml",
    # ".pre-commit-hooks.yaml",
}


#  The per-file/folder rules that determine test selection logic. Each rule has:
#  - a name (for auditing/debugging purposes)
#  - a set of path predicates (match_any) that trigger the rule if any predicate
#    matches any changed file
#  - a list of pytest paths to select if the rule is triggered (can be empty)
#  - a list of functional test scripts to select if the rule is triggered (can be empty)
CATEGORY_RULES = validate_category_rules(
    [
        # DOCS & NOTEBOOKS #
        CategoryRule(
            name="docs",
            match_any=[
                prefix("docs/"),
                all_of(suffix(".md", ".rst"), case_insensitive_match("docs")),
                all_of(suffix(".ipynb"), case_insensitive_match("docs")),
                equals("_config.yml", "_toc.yml"),
                equals(".github/workflows/build-book.yml"),
            ],
            pytest_paths=[
                # NOTE:
                # Optional docs-targeted tests may be attached here.
                # If present, docs changes will still enable the docs lane, and may also
                # contribute selections added here to the fast lane.
            ],
            functional_scripts=[
                # NOTE:
                # Optional docs-targeted functional tests may be attached here.
                # If present, docs changes will still enable the docs lane, and may also
                # contribute selections added here to the fast lane.
            ],
        ),
        CategoryRule(
            name="notebooks_examples",
            match_any=[
                prefix("examples/JUPYTER/", "examples/COLAB/"),
                all_of(suffix(".ipynb"), case_insensitive_match("examples")),
            ],
            pytest_paths=MINIMAL_PYTEST,
            functional_scripts=[],
        ),
        # CORE FUNCTIONALITY #
        CategoryRule(
            name="superanimal_modelzoo",
            match_any=[
                prefix("deeplabcut/modelzoo/"),
                case_insensitive_match("superanimal"),
                # case_insensitive_match("modelzoo"), # too broad ?
            ],
            pytest_paths=[
                "tests/test_predict_supermodel.py",
                "tests/pose_estimation_pytorch/modelzoo/",
                "tests/pose_estimation_pytorch/other/test_modelzoo.py",  # (currently all tests are skipped in this file..) # noqa: E501
            ],
            functional_scripts=[
                # TODO: decide which of these functional testscripts are useful and not too heavy # noqa: E501
                "examples/testscript_superanimal_adaptation.py",  # (runs inference + video adaptation training on shortened video) # noqa: E501
                # "examples/testscript_superanimal_create_pretrained_project.py", # (runs inference on example videos) # noqa: E501
                # "examples/testscript_superanimal_inference.py", # (runs inference on multiple videos with multiple models) # noqa: E501
                # "examples/testscript_superanimal_transfer_learning.py", # (runs full standard training pipeline after weight init) # noqa: E501
            ],
        ),
        CategoryRule(
            name="multianimal",
            match_any=[
                case_insensitive_match("multianimal"),
                all_of(prefix(POSE_TF), case_insensitive_match("multi")),
                all_of(prefix(POSE_PT), case_insensitive_match("multi")),
            ],
            pytest_paths=[
                "tests/test_auxfun_multianimal.py",
                "tests/test_pose_multianimal_imgaug.py",
                "tests/test_predict_multianimal.py",
                "tests/test_stitcher.py",
                "tests/test_trackingutils.py",
            ],
            functional_scripts=[
                "examples/testscript_tensorflow_multi_animal.py",
                "examples/testscript_pytorch_multi_animal.py",
            ],
        ),
        CategoryRule(
            name="core",
            match_any=[
                prefix(
                    "deeplabcut/core/",
                    "deeplabcut/utils/",
                    POSE_TF,
                    POSE_PT,
                ),
                equals("deeplabcut/auxiliaryfunctions.py"),
            ],
            pytest_paths=[
                "tests/test_auxiliaryfunctions.py",
                "tests/core/",
                "tests/utils/",
            ],
            functional_scripts=[
                "examples/testscript_tensorflow_single_animal.py",
                "examples/testscript_tensorflow_multi_animal.py",
                "examples/testscript_pytorch_single_animal.py",
                "examples/testscript_pytorch_multi_animal.py",
            ],
        ),
        CategoryRule(
            name="pose_estimation_tensorflow",
            match_any=[
                prefix(POSE_TF),
            ],
            pytest_paths=[
                "tests/test_dataset_augmentation.py",
                "tests/test_pose_multianimal_imgaug.py",
                "tests/test_predict_multianimal.py",
                "tests/test_evaluate.py",
                # "tests/test_inferenceutils.py",
                # "tests/test_crossvalutils.py",
            ],
            functional_scripts=[
                "examples/testscript_tensorflow_multi_animal.py",
                "examples/testscript_tensorflow_single_animal.py",
            ],
        ),
        CategoryRule(
            name="pose_estimation_pytorch",
            match_any=[
                prefix(POSE_PT),
            ],
            pytest_paths=[
                "tests/pose_estimation_pytorch/",
            ],
            functional_scripts=[
                "examples/testscript_pytorch_single_animal.py",
                "examples/testscript_pytorch_multi_animal.py",
            ],
        ),
        CategoryRule(
            name="3d_pose_estimation",
            match_any=[
                prefix("deeplabcut/pose_estimation_3d/"),
            ],
            pytest_paths=[
                "tests/test_triangulation.py",
            ],
            functional_scripts=[
                "examples/testscript_3d.py",
            ],
        ),
        CategoryRule(
            name="generate_training_dataset",
            match_any=[
                prefix("deeplabcut/generate_training_dataset/"),
            ],
            pytest_paths=[
                "tests/generate_training_dataset/",
            ],
            functional_scripts=[],
        ),
        # CI & TOOLING #
        CategoryRule(
            name="ci_workflows",
            match_any=[
                prefix(".github/workflows/"),
            ],
            pytest_paths=MINIMAL_PYTEST,
            functional_scripts=[],
        ),
        CategoryRule(
            name="ci_tools",
            match_any=[
                prefix("tools/"),
            ],
            pytest_paths=["tests/tools/"],
            functional_scripts=[],
        ),
    ]
)

CATEGORY_RULE_BY_NAME = {r.name: r for r in CATEGORY_RULES}
