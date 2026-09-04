from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_on_path():
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
