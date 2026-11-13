#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for trajectory plotting in the Analyze Videos tab."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtWidgets

import deeplabcut.gui.tabs.analyze_videos as analyze_videos_tab
from deeplabcut.gui.components import DefaultTab
from deeplabcut.gui.tabs.analyze_videos import AnalyzeVideos, AnalyzeVideosOptions

BATCHES = [(".avi", [Path("a.avi")]), (".mp4", [Path("b.mp4")])]


class FakeRoot(QtCore.QObject):
    """Minimal stand-in for MainWindow as seen by AnalyzeVideos."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._progress_bar = MagicMock()
        self.task_errors = []

    def reload_project_config(self):
        pass

    def show_task_error(self, error):
        self.task_errors.append(error)


def _options(show_trajectory_plots: bool) -> AnalyzeVideosOptions:
    return AnalyzeVideosOptions(
        config_path="config.yaml",
        shuffle=1,
        save_as_csv=False,
        filter_data=False,
        plot_trajectories=True,
        show_trajectory_plots=show_trajectory_plots,
        displayed_bodyparts=(),
        create_video_all_detections=False,
        auto_track=False,
        calibrate_assembly=False,
        assemble_with_ID_only=False,
        num_animals_in_videos=None,
        cropping=None,
        dynamic_cropping_params=(False, 0.5, 10),
        track_method=None,
    )


@pytest.fixture
def plot_calls(monkeypatch):
    """Fake DeepLabCut API where analysing the .mp4 batch fails; records plot calls."""
    calls = []

    def analyze_videos(*args, video_extensions, **kwargs):
        if video_extensions == ".mp4":
            raise ValueError("synthetic analysis failure")

    def plot_trajectories(*args, video_extensions, showfigures, **kwargs):
        calls.append((video_extensions, showfigures))

    fake_api = SimpleNamespace(analyze_videos=analyze_videos, plot_trajectories=plot_trajectories)
    monkeypatch.setattr(analyze_videos_tab, "deeplabcut", fake_api)
    return calls


@pytest.fixture
def tab(qtbot, monkeypatch):
    """An AnalyzeVideos tab without its widgets, which need a loaded project."""
    monkeypatch.setattr(DefaultTab, "_init_default_layout", lambda self: None)
    monkeypatch.setattr(AnalyzeVideos, "_set_page", lambda self: None)

    tab = AnalyzeVideos(FakeRoot(), parent=None, h1_description="")
    tab.analyze_videos_btn = QtWidgets.QPushButton()
    tab.analyze_videos_btn.setEnabled(False)
    qtbot.addWidget(tab)
    return tab


def _run_failing_pipeline(tab, options):
    tab._plot_options = options
    with pytest.raises(RuntimeError, match="synthetic analysis failure"):
        tab._run_pipeline(options, BATCHES)
    tab._handle_analysis_finished()


def test_plots_of_successful_batches_are_saved_and_shown_after_later_failure(tab, plot_calls, qtbot):
    _run_failing_pipeline(tab, _options(show_trajectory_plots=True))

    assert tab.analyze_videos_btn.isEnabled()
    qtbot.waitUntil(lambda: len(plot_calls) == 2, timeout=1000)
    assert plot_calls == [(".avi", False), (".avi", True)]


def test_plots_are_saved_but_not_shown_when_display_is_off(tab, plot_calls, qtbot):
    _run_failing_pipeline(tab, _options(show_trajectory_plots=False))

    qtbot.wait(50)
    assert plot_calls == [(".avi", False)]
