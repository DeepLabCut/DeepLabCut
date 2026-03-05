from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _repo_root() -> Path:
    # tests/intelligent_selector/conftest.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[2]


def load_selector_module() -> ModuleType:
    root = _repo_root()
    selector_path = root / "tools" / "intelligent_test_selector.py"
    if not selector_path.exists():
        raise FileNotFoundError(f"Selector script not found: {selector_path}")

    spec = importlib.util.spec_from_file_location(
        "intelligent_test_selector", selector_path
    )
    assert spec and spec.loader, "Failed to create import spec for selector"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="session")
def selector():
    """Imported selector module (tools/intelligent_test_selector.py)."""
    return load_selector_module()
