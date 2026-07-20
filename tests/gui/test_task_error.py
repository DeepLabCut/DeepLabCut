#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for MainWindow.show_task_error dialog rendering."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")


from pydantic import ValidationError
from PySide6 import QtWidgets

from deeplabcut.core.config import ProjectConfig


def _make_validation_error(cfg: dict) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        ProjectConfig.from_dict(cfg)
    return exc_info.value


class TestShowTaskError:
    @pytest.fixture
    def patched_messagebox(self, monkeypatch):
        """Intercept every QMessageBox so show_task_error never blocks."""
        captured = {}

        class _Box(QtWidgets.QMessageBox):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["instance"] = self

            def exec(self):
                captured["exec_called"] = True
                return 0

        monkeypatch.setattr(QtWidgets, "QMessageBox", _Box)
        return captured

    def test_generic_error_shows_task_failed_dialog(
        self, main_window, tmp_path, write_project_config, patched_messagebox
    ):
        """Non-config exceptions get a generic 'Task failed' dialog."""
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)
        main_window.config = str(config_path)

        main_window.show_task_error(ValueError("config is empty or null"))

        box = patched_messagebox["instance"]
        assert box.windowTitle() == "Task failed"
        assert box.text() == "DeepLabCut could not complete the task."
        assert "config is empty or null" in box.informativeText()

        button_texts = [b.text() for b in box.buttons()]
        assert "Open configuration" not in button_texts

    def test_config_validation_error_shows_open_button(
        self, main_window, tmp_path, write_project_config, patched_messagebox
    ):
        """Config validation errors show an 'Open configuration' button."""
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)
        main_window.config = str(config_path)

        error = _make_validation_error({"multianimalproject": "banana"})

        main_window.show_task_error(error, config_path=str(config_path))

        box = patched_messagebox["instance"]
        assert box.windowTitle() == "Invalid project configuration"
        assert "multianimalproject" in box.informativeText()

        button_texts = [b.text() for b in box.buttons()]
        assert "Open configuration" in button_texts
