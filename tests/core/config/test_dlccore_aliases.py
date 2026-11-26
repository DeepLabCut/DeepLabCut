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
"""The config modules moved to dlccore keep their DeepLabCut import paths."""

import importlib

import dlccore.config
import dlccore.deprecation
import pytest

import deeplabcut.core.config
from deeplabcut.core.deprecation import DLCDeprecationWarning

MOVED = ["base_config", "project_config", "validation", "versioning"]


@pytest.mark.parametrize("name", MOVED)
def test_old_path_imports_the_dlccore_module(name):
    module = importlib.import_module(f"deeplabcut.core.config.{name}")
    assert module is getattr(dlccore.config, name)
    assert getattr(deeplabcut.core.config, name) is module


def test_patching_the_old_path_reaches_dlccore(monkeypatch):
    from deeplabcut.core.config import versioning

    monkeypatch.setitem(versioning._MIGRATIONS, ("ToyConfig", 0, 1), lambda config: config)
    assert ("ToyConfig", 0, 1) in dlccore.config.versioning._MIGRATIONS


def test_deprecation_warning_is_dlccore_class():
    assert DLCDeprecationWarning is dlccore.deprecation.DLCDeprecationWarning


def test_pose_config_registers_its_migrations():
    from deeplabcut.pose_estimation_pytorch.config import pose

    migrations = dlccore.config.versioning._MIGRATIONS
    assert migrations[("PoseConfig", 0, 1)] is pose.migrate_pose_v0_to_v1
    assert migrations[("PoseConfig", 1, 0)] is pose.migrate_pose_v1_to_v0
