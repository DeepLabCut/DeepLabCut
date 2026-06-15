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
import urllib.request
from collections.abc import Callable

from PySide6 import QtCore, QtNetwork

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


def parse_version(version: str) -> tuple[int, int, int]:
    """Parses a version string into a tuple of (major, minor, patch)."""
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


class UpdateChecker(QtCore.QObject):
    finished = QtCore.Signal(object)  # emits result dict

    DLC_URL = "https://pypi.org/pypi/deeplabcut/json"
    NAPARI_DLC_URL = "https://pypi.org/pypi/napari-deeplabcut/json"

    def __init__(self, dlc_version: str, plugin_version: str, timeout_ms: int = 5000, parent=None):
        super().__init__(parent)
        self._dlc_version = dlc_version
        self._plugin_version = plugin_version
        self._timeout_ms = timeout_ms

        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._running = False
        self._silent = True
        self._replies = {}
        self._result = {}

    def is_running(self) -> bool:
        return self._running

    def check(self, silent: bool = True):
        if self._running:
            # if a manual check happens while a silent one is running,
            # keep the in-flight request but upgrade the result visibility
            self._silent = self._silent and silent
            return

        self._running = True
        self._silent = silent
        self._result = {
            "silent": silent,
            "is_latest": True,
            "latest_version": None,
            "is_latest_plugin": True,
            "latest_plugin_version": None,
            "error": None,
        }

        self._start_request("dlc", self.DLC_URL)
        self._start_request("napari-dlc", self.NAPARI_DLC_URL)
        self._timer.start(self._timeout_ms)

    def cancel(self):
        if not self._running:
            return
        self._silent = True
        self._abort_all()
        self._finish()

    def _start_request(self, key: str, url: str):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        req.setHeader(
            QtNetwork.QNetworkRequest.KnownHeaders.UserAgentHeader,
            "DeepLabCut GUI UpdateChecker",
        )

        reply = self._manager.get(req)
        self._replies[key] = reply
        reply.finished.connect(lambda key=key, reply=reply: self._on_reply_finished(key, reply))

    def _on_reply_finished(self, key: str, reply: QtNetwork.QNetworkReply):
        try:
            if reply.error() != QtNetwork.QNetworkReply.NetworkError.NoError:
                # keep the first network-ish error but remain non-fatal overall
                if self._result["error"] is None:
                    self._result["error"] = reply.errorString()
                return

            payload = bytes(reply.readAll())
            latest_version = json.loads(payload.decode("utf-8"))["info"]["version"]

            if key == "dlc":
                self._result["latest_version"] = latest_version
                self._result["is_latest"] = self._is_up_to_date(self._dlc_version, latest_version)
            else:
                self._result["latest_plugin_version"] = latest_version
                self._result["is_latest_plugin"] = self._is_up_to_date(self._plugin_version, latest_version)
        except Exception as e:
            if self._result["error"] is None:
                self._result["error"] = str(e)
        finally:
            reply.deleteLater()
            self._replies.pop(key, None)

            if self._running and not self._replies:
                self._finish()

    def _on_timeout(self):
        if not self._running:
            return
        if self._result["error"] is None:
            self._result["error"] = "Update check timed out."
        self._abort_all()
        self._finish()

    def _abort_all(self):
        for reply in list(self._replies.values()):
            if reply is not None and reply.isRunning():
                reply.abort()
            reply.deleteLater()
        self._replies.clear()

    def _finish(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        self._result["silent"] = self._silent
        self.finished.emit(self._result)

    @staticmethod
    def _is_up_to_date(installed: str, latest: str) -> bool:
        if Version is not None:
            try:
                return Version(installed) >= Version(latest)
            except InvalidVersion:
                return installed == latest
        return installed == latest
