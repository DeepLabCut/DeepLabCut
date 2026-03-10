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
import json
import re
import socket
import urllib.request
from typing import Callable, Tuple
from urllib.error import URLError

from PySide6 import QtCore

try:
    from packaging.version import InvalidVersion, Version
except Exception:  # packaging should usually be available, but keep fallback safe
    Version = None
    InvalidVersion = Exception


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


def parse_version(version: str) -> Tuple[int, int, int]:
    """
    Parses a version string into a tuple of (major, minor, patch).
    """
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return tuple(int(part) for part in match.groups())
    else:
        raise ValueError(f"Invalid version format: {version}")


def check_pypi_version(package_name: str, installed_version: str, timeout: float = 5.0):
    """
    Return (is_latest, latest_version) for a package on PyPI.

    - Uses a real network timeout via urllib.
    - Treats locally newer/dev versions as up-to-date when packaging is available.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"

    with urllib.request.urlopen(url, timeout=timeout) as response:
        contents = response.read()

    latest_version = json.loads(contents)["info"]["version"]

    if Version is not None:
        try:
            is_latest = Version(installed_version) >= Version(latest_version)
        except InvalidVersion:
            is_latest = installed_version == latest_version
    else:
        is_latest = installed_version == latest_version

    return is_latest, latest_version


def is_latest_deeplabcut_version(timeout: float = 5.0):
    from deeplabcut import __version__

    return check_pypi_version("deeplabcut", __version__, timeout=timeout)


def is_latest_plugin_version(timeout: float = 5.0):
    from napari_deeplabcut import __version__

    return check_pypi_version("napari-deeplabcut", __version__, timeout=timeout)


class UpdateCheckWorker(QtCore.QObject):
    finished = QtCore.Signal(object)  # emits a dict

    def __init__(self, timeout=2.0, parent=None):
        super().__init__(parent)
        self.timeout = timeout

    @QtCore.Slot()
    def run(self):
        result = {
            "is_latest": True,
            "latest_version": None,
            "is_latest_plugin": True,
            "latest_plugin_version": None,
            "error": None,
        }

        try:
            (
                result["is_latest"],
                result["latest_version"],
            ) = is_latest_deeplabcut_version(timeout=self.timeout)
            # NOTE: @C-Achard 2026-03-10 If we ever make the plugin optional,
            # this will need to be adapted.
            # For now since it is a hard dep of the GUI, we keep it in this try block
            (
                result["is_latest_plugin"],
                result["latest_plugin_version"],
            ) = is_latest_plugin_version(timeout=self.timeout)
        except (URLError, socket.timeout, TimeoutError, OSError) as e:
            # Connectivity issues should stay non-fatal / silent
            result["error"] = e
        except Exception as e:
            # Unexpected failures also go back to the GUI thread for logging if desired
            result["error"] = e

        self.finished.emit(result)
