#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for worker-thread task execution and error marshalling."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from deeplabcut.gui.utils import CaptureWorker, Worker, move_to_separate_thread


def _boom():
    raise ZeroDivisionError("boom")


def test_worker_emits_finished_on_success(qtbot):
    calls = []
    worker = Worker(lambda: calls.append(True))

    with qtbot.waitSignal(worker.finished, timeout=1000):
        worker.run()

    assert calls == [True]


def test_worker_emits_error_and_finished_on_exception(qtbot):
    """Regression: a raising task used to kill the worker without any signal,
    leaving progress bars spinning and buttons disabled forever."""
    worker = Worker(_boom)
    errors = []
    worker.error.connect(errors.append)

    with qtbot.waitSignal(worker.finished, timeout=1000):
        worker.run()

    assert len(errors) == 1
    assert isinstance(errors[0], ZeroDivisionError)


def test_capture_worker_captures_outputs():
    worker = CaptureWorker(lambda: "result")
    worker.run()
    assert worker.outputs == "result"


def test_capture_worker_outputs_none_on_error(qtbot):
    worker = CaptureWorker(_boom)
    errors = []
    worker.error.connect(errors.append)

    with qtbot.waitSignal(worker.finished, timeout=1000):
        worker.run()

    assert worker.outputs is None
    assert len(errors) == 1


def test_thread_quits_after_exception(qtbot):
    """Regression: the QThread must stop even when the task raises."""
    worker, thread = move_to_separate_thread(_boom)
    errors = []
    worker.error.connect(errors.append)

    with qtbot.waitSignal(thread.finished, timeout=3000):
        thread.start()

    qtbot.waitUntil(thread.isFinished, timeout=3000)
    assert len(errors) == 1
    assert isinstance(errors[0], ZeroDivisionError)
