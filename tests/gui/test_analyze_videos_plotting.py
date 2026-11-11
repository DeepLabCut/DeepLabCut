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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

import deeplabcut.gui.tabs.analyze_videos as analyze_videos_tab
from deeplabcut.gui.tabs.analyze_videos import AnalyzeVideos, AnalyzeVideosOptions, TrajectoryPlotsDialog

BATCHES = [(".avi", [Path("a.avi")]), (".mp4", [Path("b.mp4")])]


class FakeRoot(QtCore.QObject):
    """Minimal stand-in for MainWindow as seen by AnalyzeVideos."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._progress_bar = MagicMock()
        self.is_multianimal = False
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
def fake_api(monkeypatch):
    """Fake DeepLabCut API that records plot calls.

    Set ``failing_extension`` to make analysis fail, or ``nothing_to_plot`` to plot no figures.
    """
    api = SimpleNamespace(plot_calls=[], failing_extension=None, nothing_to_plot=False)

    def analyze_videos(*args, video_extensions, **kwargs):
        if video_extensions == api.failing_extension:
            raise ValueError("synthetic analysis failure")

    def plot_trajectories(*args, videos, video_extensions, showfigures, return_figures=False, **kwargs):
        api.plot_calls.append((video_extensions, showfigures, return_figures))
        if return_figures:
            plotted = [] if api.nothing_to_plot else videos
            return {Path("plot-poses") / Path(v).stem / "trajectory.png": Figure() for v in plotted}

    api.analyze_videos = analyze_videos
    api.plot_trajectories = plot_trajectories
    monkeypatch.setattr(analyze_videos_tab, "deeplabcut", api)
    return api


@pytest.fixture
def shown_dialogs(monkeypatch):
    """Records the plots each dialog would display, and whether it was built on the GUI thread."""
    shown = []

    class FakeDialog:
        def __init__(self, plot_figures, parent=None):
            is_gui_thread = QtCore.QThread.currentThread() is QtWidgets.QApplication.instance().thread()
            shown.append((plot_figures, is_gui_thread))

        def show(self):
            pass

    monkeypatch.setattr(analyze_videos_tab, "TrajectoryPlotsDialog", FakeDialog)
    return shown


@pytest.fixture
def run_analysis(qtbot, monkeypatch):
    """Runs ``AnalyzeVideos.analyze_videos`` on BATCHES and waits for its worker thread to finish."""
    monkeypatch.setattr(AnalyzeVideos, "_set_page", lambda self: None)

    tab = AnalyzeVideos(FakeRoot(), parent=None, h1_description="")
    tab.analyze_videos_btn = QtWidgets.QPushButton()
    tab._get_video_batches = lambda: BATCHES
    qtbot.addWidget(tab)

    def _run(show_trajectory_plots=True):
        tab._collect_options = lambda: _options(show_trajectory_plots)
        tab.analyze_videos()
        qtbot.waitUntil(lambda: tab.thread.isFinished() and tab.analyze_videos_btn.isEnabled(), timeout=5000)
        qtbot.wait(10)  # let queued slots run
        return tab

    return _run


def test_plots_are_drawn_once_on_worker_and_shown_on_gui_thread(run_analysis, fake_api, shown_dialogs):
    run_analysis()

    assert fake_api.plot_calls == [(".avi", False, True), (".mp4", False, True)]
    [(plot_figures, is_gui_thread)] = shown_dialogs
    assert list(plot_figures) == [Path("plot-poses/a/trajectory.png"), Path("plot-poses/b/trajectory.png")]
    assert is_gui_thread


def test_plots_are_not_shown_when_display_is_off(run_analysis, fake_api, shown_dialogs):
    run_analysis(show_trajectory_plots=False)

    assert fake_api.plot_calls == [(".avi", False, False), (".mp4", False, False)]
    assert shown_dialogs == []


def test_plots_are_not_shown_when_analysis_fails(run_analysis, fake_api, shown_dialogs):
    fake_api.failing_extension = ".mp4"

    tab = run_analysis()

    assert len(tab.root.task_errors) == 1
    assert fake_api.plot_calls == [(".avi", False, True)]
    assert shown_dialogs == []


def test_display_error_is_reported_to_the_user(run_analysis, fake_api, monkeypatch):
    class FailingDialog(TrajectoryPlotsDialog):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("synthetic display failure")

    monkeypatch.setattr(analyze_videos_tab, "TrajectoryPlotsDialog", FailingDialog)

    tab = run_analysis()

    assert [str(error) for error in tab.root.task_errors] == ["synthetic display failure"]


def _open_plot_windows(tab):
    return [window for window in tab.findChildren(TrajectoryPlotsDialog) if window.isVisible()]


@pytest.fixture
def first_plot_window(run_analysis, fake_api):
    """The tab and the plot window left open by a first, successful run."""
    tab = run_analysis()
    [window] = _open_plot_windows(tab)
    return tab, window


@pytest.mark.parametrize("user_closed_it", [False, True], ids=["still-open", "closed-by-user"])
def test_new_plots_replace_the_previous_plot_window(run_analysis, first_plot_window, user_closed_it):
    tab, first_window = first_plot_window
    if user_closed_it:
        first_window.close()

    run_analysis()

    [window] = _open_plot_windows(tab)
    assert window is not first_window


@pytest.mark.parametrize(
    "show_trajectory_plots, api_changes",
    [
        pytest.param(False, {}, id="display-off"),
        pytest.param(True, {"failing_extension": ".mp4"}, id="analysis-fails"),
        pytest.param(True, {"nothing_to_plot": True}, id="nothing-to-plot"),
    ],
)
def test_previous_plot_window_stays_open_when_no_new_plots_arrive(
    run_analysis, fake_api, first_plot_window, show_trajectory_plots, api_changes
):
    tab, first_window = first_plot_window
    vars(fake_api).update(api_changes)

    run_analysis(show_trajectory_plots)

    assert _open_plot_windows(tab) == [first_window]


def _figure_with_colorbar():
    """A figure built like those of ``PlottingResults``: colorbar added without a layout engine."""
    fig = Figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("Frame Index")
    fig.colorbar(ScalarMappable(), ax=ax)
    return fig


def test_dialog_shows_one_tab_per_video_with_a_toolbar_per_figure(qtbot):
    plot_figures = {
        Path("plot-poses") / video / name: _figure_with_colorbar()
        for video in ("a", "b")
        for name in ("trajectory.png", "hist.png")
    }

    dialog = TrajectoryPlotsDialog(plot_figures)
    qtbot.addWidget(dialog)

    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["a", "b"]
    first_tab = tabs.widget(0).widget()
    first_tab_figures = [canvas.figure for canvas in first_tab.findChildren(FigureCanvas)]
    assert first_tab_figures == list(plot_figures.values())[:2]
    assert len(first_tab.findChildren(NavigationToolbar2QT)) == 2


def test_dialog_draws_figures_with_visible_labels(qtbot):
    figure = _figure_with_colorbar()
    dialog = TrajectoryPlotsDialog({Path("plot-poses/a/plot.png"): figure})
    qtbot.addWidget(dialog)

    canvas = dialog.findChild(FigureCanvas)
    canvas.draw()

    xlabel = figure.axes[0].xaxis.label.get_window_extent(canvas.get_renderer())
    assert xlabel.y0 >= 0  # Label "Frame Index" is visible
