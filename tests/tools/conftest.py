from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _repo_root() -> Path:
    # tests/tools/conftest.py -> repo root is 2 levels up
    return Path(__file__).resolve().parents[2]


def load_selector_module() -> ModuleType:
    root = _repo_root()
    tools_dir = root / "tools"
    init_file = tools_dir / "__init__.py"
    selector_path = tools_dir / "test_selector.py"

    if not selector_path.exists():
        raise FileNotFoundError(f"Selector script not found: {selector_path}")

    if not init_file.exists():
        raise FileNotFoundError(f"tools package marker not found: {init_file}")

    # Ensure repo root is importable so `tools.test_selector` resolves as a package import.
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return importlib.import_module("tools.test_selector")


@pytest.fixture(scope="session")
def selector():
    """Imported selector module (tools.test_selector)."""
    return load_selector_module()
