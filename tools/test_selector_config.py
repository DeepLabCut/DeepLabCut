"""Test selector configuration utilities."""
from typing import Callable

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
# Configuration (simple, auditable)
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


CATEGORY_RULES = [
    {
        "name": "docs",
        "match_any": [
            prefix("docs/"),
            all_of(suffix(".md", ".rst"), case_insensitive_match("docs")),
            all_of(suffix(".ipynb"), case_insensitive_match("docs")),
            equals("_config.yml", "_toc.yml"),
            equals(".github/workflows/build-book.yml"),
        ],
        "pytest_paths": [
            # NOTE:
            # Optional docs-targeted tests may be attached here.
            # If present, docs changes will still enable the docs lane, and may also
            # contribute selections to the fast lane.
        ],
        "functional_scripts": [
            # NOTE:
            # Optional docs-targeted functional tests may be attached here.
            # If present, docs changes will still enable the docs lane, and may also
            # contribute selections to the fast lane.
        ],
    },
    {
        "name": "notebooks_examples",
        "match_any": [
            prefix("examples/JUPYTER/", "examples/COLAB/"),
            all_of(suffix(".ipynb"), case_insensitive_match("examples")),
        ],
        "pytest_paths": MINIMAL_PYTEST,
        "functional_scripts": [],
    },
    {
        "name": "superanimal_modelzoo",
        "match_any": [
            prefix("deeplabcut/modelzoo/"),
            case_insensitive_match("superanimal"),
            # case_insensitive_match("modelzoo"), # too broad ?
        ],
        "pytest_paths": [
            "tests/test_predict_supermodel.py",
            "tests/pose_estimation_pytorch/modelzoo/",
        ],
        "functional_scripts": [],
    },
    {
        "name": "multianimal",
        "match_any": [
            case_insensitive_match("multianimal"),
            all_of(prefix(POSE_TF), case_insensitive_match("multi")),
            all_of(prefix(POSE_PT), case_insensitive_match("multi")),
        ],
        "pytest_paths": [
            "tests/test_auxfun_multianimal.py",
            "tests/test_pose_multianimal_imgaug.py",
            "tests/test_predict_multianimal.py",
        ],
        "functional_scripts": [
            "examples/testscript_multianimal.py",
        ],
    },
    {
        "name": "core",
        "match_any": [
            prefix(
                "deeplabcut/core/",
                "deeplabcut/utils/",
                POSE_TF,
                POSE_PT,
            ),
            equals("deeplabcut/auxiliaryfunctions.py"),
        ],
        "pytest_paths": [
            "tests/test_auxiliaryfunctions.py",
            "tests/core/",
            "tests/utils/",
            "tests/pose_estimation_pytorch/",
        ],
        "functional_scripts": [],
    },
    {
        "name": "ci_workflows",
        "match_any": [
            prefix(".github/workflows/"),
        ],
        "pytest_paths": MINIMAL_PYTEST,
        "functional_scripts": [],
    },
    {
        "name": "ci_tools",
        "match_any": [
            prefix("tools/"),
        ],
        "pytest_paths": ["tests/tools/"],
        "functional_scripts": [],
    },
]
