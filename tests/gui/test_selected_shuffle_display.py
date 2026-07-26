#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the shuffle info display degrading gracefully on bad shuffles."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

import PySide6.QtCore as QtCore

from deeplabcut.core.engine import Engine
from deeplabcut.gui.displays.selected_shuffle_display import SelectedShuffleDisplay


class FakeRoot(QtCore.QObject):
    """Minimal stand-in for MainWindow as seen by SelectedShuffleDisplay."""

    shuffle_change = QtCore.Signal(int)
    engine_change = QtCore.Signal(object)
    shuffle_created = QtCore.Signal(int)

    def __init__(self, pose_cfg_path: str, raise_error: Exception | None = None):
        super().__init__()
        self.shuffle_value = 1
        self.engine = Engine.PYTORCH
        self._pose_cfg_path = pose_cfg_path
        self._raise_error = raise_error

    @property
    def pose_cfg_path(self) -> str:
        if self._raise_error is not None:
            raise self._raise_error
        return self._pose_cfg_path


def test_missing_pose_cfg_shows_error_instead_of_crashing(qtbot, tmp_path):
    missing = tmp_path / "missing" / "pose_cfg.yaml"
    display = SelectedShuffleDisplay(FakeRoot(str(missing)))
    qtbot.addWidget(display)

    assert display.pose_cfg is None
    assert "was not created" in display._label.text()


def test_unresolvable_shuffle_shows_error(qtbot, tmp_path):
    root = FakeRoot(str(tmp_path), raise_error=ValueError("no such shuffle"))
    display = SelectedShuffleDisplay(root)
    qtbot.addWidget(display)

    assert display.pose_cfg is None
    assert "Failed to read shuffle 1" in display._label.text()


def test_valid_pose_cfg_is_displayed(qtbot, tmp_path):
    pose_cfg_path = tmp_path / "pytorch_config.yaml"
    pose_cfg_path.write_text("net_type: resnet_50\nmethod: TD\n")

    display = SelectedShuffleDisplay(FakeRoot(str(pose_cfg_path)))
    qtbot.addWidget(display)

    assert display.pose_cfg == {"net_type": "resnet_50", "method": "TD"}
    assert "resnet_50" in display._label.text()
    assert "top-down" in display._label.text()


def test_pose_cfg_without_method_key_does_not_crash(qtbot, tmp_path):
    """Regression: pose_cfg.get('method').lower() raised AttributeError."""
    pose_cfg_path = tmp_path / "pytorch_config.yaml"
    pose_cfg_path.write_text("net_type: resnet_50\n")

    display = SelectedShuffleDisplay(FakeRoot(str(pose_cfg_path)))
    qtbot.addWidget(display)

    assert display.pose_cfg == {"net_type": "resnet_50"}
    assert "top-down" not in display._label.text()


def test_pose_cfg_signal_carries_none(qtbot, tmp_path):
    """Regression: the signal was declared Signal(dict) and mangled None."""
    missing = tmp_path / "missing" / "pose_cfg.yaml"
    display = SelectedShuffleDisplay(FakeRoot(str(missing)))
    qtbot.addWidget(display)

    received = []
    display.pose_cfg_signal.connect(received.append)
    display.pose_cfg = None

    assert received == [None]
