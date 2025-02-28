#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from typing import Callable

from PySide6 import QtCore


class Worker(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        self.func()
        self.finished.emit()


class CaptureWorker(Worker):
    """A worker that captures outputs from methods that are run."""

    def __init__(self, func: Callable):
        super().__init__(func)
        self.outputs = None

    def run(self):
        self.outputs = self.func()
        self.finished.emit()


def move_to_separate_thread(func: Callable, capture_outputs: bool = False):
    thread = QtCore.QThread()
    if capture_outputs:
        worker = CaptureWorker(func)
    else:
        worker = Worker(func)

    worker.finished.connect(worker.deleteLater)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def stop_thread():
        thread.quit()
        thread.wait()

    worker.finished.connect(stop_thread)
    return worker, thread


def is_latest_deeplabcut_version():
    import json
    import urllib.request
    from deeplabcut import VERSION

    url = "https://pypi.org/pypi/deeplabcut/json"
    contents = urllib.request.urlopen(url).read()
    latest_version = json.loads(contents)["info"]["version"]
    return VERSION == latest_version, latest_version
