#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Shared fixtures for training-dataset tests."""

from pathlib import Path

import pytest

from deeplabcut.core.config import ProjectConfig


@pytest.fixture
def valid_project(tmp_path: Path) -> Path:
    """Create a minimal project: a canonical config.yaml, and empty videos/ and labeled-data/."""

    (tmp_path / "videos").mkdir()
    (tmp_path / "labeled-data").mkdir()

    ProjectConfig(
        Task="test-project",
        scorer="test-scorer",
        date="2026-01-01",
        project_path=tmp_path,
        bodyparts=["nose", "tail"],
    ).to_yaml(tmp_path / "config.yaml")

    return tmp_path
