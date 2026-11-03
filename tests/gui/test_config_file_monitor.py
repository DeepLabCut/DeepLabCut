#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the external-edit watcher on the project configuration."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6 import QtWidgets

from deeplabcut.gui.config_file_monitor import ConfigFileMonitor


@pytest.fixture
def status_bar(qtbot):
    bar = QtWidgets.QStatusBar()
    qtbot.addWidget(bar)
    return bar


def _make_monitor(status_bar, config_path, on_reload=lambda: True):
    monitor = ConfigFileMonitor(status_bar, on_reload=on_reload, debounce_ms=50)
    monitor.set_path(str(config_path))
    monitor.mark_current()
    return monitor


def test_external_edit_offers_reload(qtbot, tmp_path, status_bar):
    config = tmp_path / "config.yaml"
    config.write_text("Task: test\n")
    monitor = _make_monitor(status_bar, config)
    assert monitor._reload_button.isHidden()

    config.write_text("Task: test\nedited: true\n")

    qtbot.waitUntil(lambda: not monitor._reload_button.isHidden(), timeout=5000)


def test_mark_current_dismisses_reload_offer(qtbot, tmp_path, status_bar):
    config = tmp_path / "config.yaml"
    config.write_text("Task: test\n")
    monitor = _make_monitor(status_bar, config)

    config.write_text("Task: changed\n")
    qtbot.waitUntil(lambda: not monitor._reload_button.isHidden(), timeout=5000)

    monitor.mark_current()  # simulates a successful reload
    assert monitor._reload_button.isHidden()


def test_set_path_resets_pending_notification(qtbot, tmp_path, status_bar):
    config = tmp_path / "config.yaml"
    config.write_text("Task: test\n")
    monitor = _make_monitor(status_bar, config)

    config.write_text("Task: changed\n")
    qtbot.waitUntil(lambda: not monitor._reload_button.isHidden(), timeout=5000)

    other = tmp_path / "other.yaml"
    other.write_text("Task: other\n")
    monitor.set_path(str(other))

    assert monitor._reload_button.isHidden()


def test_reload_button_triggers_callback(qtbot, tmp_path, status_bar):
    config = tmp_path / "config.yaml"
    config.write_text("Task: test\n")
    reloads = []
    monitor = _make_monitor(status_bar, config, on_reload=lambda: reloads.append(True))

    monitor._reload_button.click()

    assert reloads == [True]
