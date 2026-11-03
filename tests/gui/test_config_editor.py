#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the raw YAML tree editor (ConfigEditor)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from deeplabcut.gui.widgets import ConfigEditor
from deeplabcut.utils import auxiliaryfunctions


def test_editor_displays_project_config_as_tree(qtbot, tmp_path, write_project_config):
    """Regression: read_config returns a ProjectConfig model, which the tree
    editor cannot display; the editor must read the raw YAML dict instead."""
    config_path = tmp_path / "config.yaml"
    write_project_config(config_path, tmp_path)

    editor = ConfigEditor(str(config_path))
    qtbot.addWidget(editor)

    assert isinstance(editor.cfg, dict)

    root = editor.viewer.tree.invisibleRootItem()
    keys = {root.child(i).text(0) for i in range(root.childCount())}
    assert {"Task", "bodyparts", "multianimalproject"} <= keys


def test_editor_opens_config_that_fails_validation(qtbot, tmp_path, write_project_config):
    """The editor is a recovery tool: it must open invalid configs so the
    user can repair them."""
    config_path = tmp_path / "config.yaml"
    write_project_config(config_path, tmp_path, extra="not_a_dlc_setting: 1")

    editor = ConfigEditor(str(config_path))  # must not raise
    qtbot.addWidget(editor)

    assert editor.cfg["not_a_dlc_setting"] == 1


def test_editor_saves_edits_back_to_disk(qtbot, tmp_path, write_project_config):
    config_path = tmp_path / "config.yaml"
    write_project_config(config_path, tmp_path)

    editor = ConfigEditor(str(config_path))
    qtbot.addWidget(editor)
    editor.cfg["colormap"] = "viridis"
    editor.accept()

    on_disk = auxiliaryfunctions.read_plainconfig(str(config_path))
    assert on_disk["colormap"] == "viridis"
