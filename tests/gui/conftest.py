#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Shared fixtures for GUI tests (headless Qt via pytest-qt)."""

import os
import sys

# Render off-screen so tests do not open windows; must be set before Qt loads.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")


@pytest.fixture
def write_project_config():
    """Writer for a minimal valid project config.yaml.

    ``project_path`` should be the parent of ``config_path`` so that
    ``read_config`` does not repair (and rewrite) the file.
    """

    def _write(config_path, project_path, task: str = "demo", extra: str = "") -> None:
        lines = [
            f"Task: {task}",
            "scorer: tester",
            "date: Jan1",
            "multianimalproject: false",
            f"project_path: {project_path.as_posix()}",
            "bodyparts: [nose, tail]",
            "video_sets: {}",
        ]
        if extra:
            lines.append(extra)
        config_path.write_text("\n".join(lines) + "\n")

    return _write


@pytest.fixture
def main_window(qapp, monkeypatch):
    """A real MainWindow, constructed off-screen and torn down cleanly."""
    from PySide6 import QtWidgets

    from deeplabcut.gui.window import MainWindow

    # Keep QSettings written by save_settings() out of the user's real settings.
    qapp.setOrganizationName("DeepLabCut-Tests")
    qapp.setApplicationName("DLC-GUI-Tests")

    # closeEvent pops a blocking confirmation dialog; auto-confirm it in tests.
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )

    original_stdout = sys.stdout  # MainWindow redirects stdout to its writer
    window = MainWindow(qapp)
    try:
        yield window
    finally:
        window.close()
        sys.stdout = original_stdout
