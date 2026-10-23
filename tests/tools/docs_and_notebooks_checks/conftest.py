from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

TOOL_PATH = Path(__file__).resolve().parents[3] / "tools" / "docs_and_notebooks_check.py"


# -----------------------------
# Module loader (tools/ is not necessarily a package)
# -----------------------------
def load_tool_module() -> ModuleType:
    assert TOOL_PATH.exists(), f"Missing tool: {TOOL_PATH}"

    spec = importlib.util.spec_from_file_location("docs_and_notebooks_check", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="session")
def tool() -> ModuleType:
    return load_tool_module()
