#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Watch the active project configuration file for external edits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtWidgets


class ConfigFileMonitor(QtCore.QObject):
    """Offer an explicit reload when the active config file changes on disk."""

    def __init__(
        self,
        status_bar: QtWidgets.QStatusBar,
        on_reload: Callable[[], bool],
        parent: QtCore.QObject | None = None,
        debounce_ms: int = 300,
    ) -> None:
        super().__init__(parent)
        self._path: str | None = None
        self._loaded_signature: tuple[int, int] | None = None

        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watcher.directoryChanged.connect(self._on_file_changed)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._check_for_change)

        self._reload_button = QtWidgets.QPushButton("Reload configuration")
        self._reload_button.clicked.connect(on_reload)
        self._reload_button.hide()
        status_bar.addPermanentWidget(self._reload_button)
        self._status_bar = status_bar

    def set_path(self, path: str | None) -> None:
        """Watch ``path`` and its parent directory, clearing any previous watches."""
        watched_files = self._watcher.files()
        if watched_files:
            self._watcher.removePaths(watched_files)
        watched_dirs = self._watcher.directories()
        if watched_dirs:
            self._watcher.removePaths(watched_dirs)

        self._path = path
        self._loaded_signature = None
        self._debounce.stop()
        self._reload_button.hide()

        if path is not None:
            config_path = Path(path)
            if config_path.is_file():
                self._watcher.addPath(path)
            # Watch the parent directory so atomic file replacements
            # (delete + rename) are also detected.
            parent = str(config_path.parent)
            if parent and Path(parent).is_dir():
                self._watcher.addPath(parent)

    def mark_current(self) -> None:
        """Record that the in-memory config matches the file currently on disk."""
        self._loaded_signature = self._signature()
        if self._path is not None and Path(self._path).is_file():
            if self._path not in self._watcher.files():
                self._watcher.addPath(self._path)
        self._reload_button.hide()

    def _signature(self) -> tuple[int, int] | None:
        if self._path is None:
            return None
        try:
            stat = Path(self._path).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _on_file_changed(self, _path: str) -> None:
        self._debounce.start()

    def _check_for_change(self) -> None:
        if self._path is None or self._loaded_signature is None:
            return

        current = self._signature()
        if current == self._loaded_signature:
            return

        # Some editors replace the file, which drops the watch path.
        # Watching the parent directory helps detect such replacements.
        if current is not None and self._path not in self._watcher.files():
            self._watcher.addPath(self._path)

        self._status_bar.showMessage("Project configuration changed on disk.")
        self._reload_button.show()
