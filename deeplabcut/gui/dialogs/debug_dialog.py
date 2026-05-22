#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

from __future__ import annotations

from collections.abc import Callable, Iterable

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFontDatabase, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deeplabcut.core.debug import (
    ExecutableSpec,
    InMemoryDebugRecorder,
    LibrarySpec,
    build_debug_report,
    get_debug_recorder,
    install_debug_recorder,
)


def make_log_text_provider(
    *,
    recorder: InMemoryDebugRecorder | None,
    limit: int = 300,
) -> Callable[[], str]:
    """Return a callable that renders recent captured logs."""

    def _provider() -> str:
        if recorder is None:
            return "<debug recorder unavailable>"
        return recorder.render_text(limit=limit)

    return _provider


def make_issue_report_provider(
    *,
    recorder: InMemoryDebugRecorder | None,
    libraries: Iterable[LibrarySpec | str] | None = None,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_module_paths: bool = False,
    include_executable_paths: bool = True,
    log_limit: int = 300,
) -> Callable[[], str]:
    """Return a callable that builds a full DLC debug report.

    ``libraries`` and ``executables`` are normalized to tuples so the returned
    provider can be called repeatedly even if the caller passed a generator or
    another one-shot iterable.
    """
    libraries_snapshot = None if libraries is None else tuple(libraries)
    executables_snapshot = None if executables is None else tuple(executables)

    def _provider() -> str:
        return build_debug_report(
            recorder=recorder,
            libraries=libraries_snapshot,
            executables=executables_snapshot,
            include_module_paths=include_module_paths,
            include_executable_paths=include_executable_paths,
            log_limit=log_limit,
        )

    return _provider


class DebugTextDialog(QDialog):
    """
    Minimal, application-agnostic debug text viewer.

    This widget only knows how to:
    - fetch text from a callable
    - display it read-only
    - copy it to clipboard
    - refresh it on demand

    It intentionally knows nothing about:
    - recorder internals
    - DLC main window internals
    - environment/report formatting
    """

    def __init__(
        self,
        *,
        title: str,
        text_provider: Callable[[], str],
        parent: QWidget | None = None,
        initial_hint: str = "Read-only diagnostic output",
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(950, 700)

        self._text_provider = text_provider

        self._build_ui(initial_hint=initial_hint)

    def update_content(
        self,
        *,
        title: str | None = None,
        text_provider: Callable[[], str] | None = None,
        hint: str | None = None,
    ) -> None:
        """Update dialog metadata when reusing an existing instance."""
        if title is not None:
            self.setWindowTitle(title)
        if text_provider is not None:
            self._text_provider = text_provider
        if hint is not None:
            self._hint_label.setText(hint)

    def _build_ui(self, *, initial_hint: str) -> None:
        layout = QVBoxLayout(self)

        self._hint_label = QLabel(initial_hint, self)
        self._hint_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._hint_label)

        self._text_edit = QPlainTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Use a fixed-width system font for logs / reports
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._text_edit.setFont(font)

        layout.addWidget(self._text_edit, stretch=1)

        button_row = QHBoxLayout()

        self._status_label = QLabel("", self)
        self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        button_row.addWidget(self._status_label, stretch=1)

        self._refresh_btn = QPushButton("Refresh", self)
        self._refresh_btn.clicked.connect(self.refresh_text)
        button_row.addWidget(self._refresh_btn)

        self._copy_btn = QPushButton("Copy to clipboard", self)
        self._copy_btn.clicked.connect(self.copy_to_clipboard)
        button_row.addWidget(self._copy_btn)

        self._close_btn = QPushButton("Close", self)
        self._close_btn.clicked.connect(self.close)
        button_row.addWidget(self._close_btn)

        layout.addLayout(button_row)

        # Optional keyboard shortcut
        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_to_clipboard)
        self.addAction(copy_action)

    def refresh_text(self) -> None:
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            text = self._text_provider()
        except Exception as exc:
            text = f"[debug-dialog] failed to build debug text\n\n{exc!r}"
        finally:
            QApplication.restoreOverrideCursor()

        self._text_edit.setPlainText(text or "<no debug text available>")
        self._text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        self._status_label.setText("")

    def copy_to_clipboard(self) -> None:
        try:
            text = self._text_edit.toPlainText()
            QApplication.clipboard().setText(text)
            self._status_label.setText("Copied to clipboard")
        except Exception:
            self._status_label.setText("Could not copy to clipboard")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Refresh each time the dialog becomes visible."""
        super().showEvent(event)
        self.refresh_text()


def _get_or_create_debug_dialog(
    *,
    parent: QWidget,
    title: str,
    text_provider: Callable[[], str],
    text_hint: str,
    attr_name: str = "_dlc_debug_dialog",
) -> DebugTextDialog:
    """
    Reuse a single dialog instance attached to ``parent``.

    Storing the dialog on the main window avoids accidental garbage collection
    and prevents opening a pile of duplicate windows.
    """
    dlg = getattr(parent, attr_name, None)
    if isinstance(dlg, DebugTextDialog):
        dlg.update_content(
            title=title,
            text_provider=text_provider,
            hint=text_hint,
        )
        return dlg

    dlg = DebugTextDialog(
        title=title,
        text_provider=text_provider,
        parent=parent,
        initial_hint=text_hint,
    )
    setattr(parent, attr_name, dlg)
    return dlg


def show_debug_report_dialog(
    *,
    parent: QWidget,
    recorder: InMemoryDebugRecorder | None = None,
    logger_name: str = "deeplabcut",
    libraries: Iterable[LibrarySpec | str] | None = None,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_module_paths: bool = False,
    include_executable_paths: bool = True,
    log_limit: int = 300,
    dialog_attr_name: str = "_dlc_debug_dialog",
) -> DebugTextDialog:
    """
    Open (or reuse) the full diagnostic report dialog.

    If ``recorder`` is not provided, this function tries to reuse an existing
    recorder for the given logger namespace and installs one if missing.
    """
    if recorder is None:
        recorder = get_debug_recorder(logger_name=logger_name)
        if recorder is None:
            recorder = install_debug_recorder(logger_name=logger_name)

    provider = make_issue_report_provider(
        recorder=recorder,
        libraries=libraries,
        executables=executables,
        include_module_paths=include_module_paths,
        include_executable_paths=include_executable_paths,
        log_limit=log_limit,
    )

    dlg = _get_or_create_debug_dialog(
        parent=parent,
        title="DeepLabCut debug log",
        text_provider=provider,
        text_hint=("Diagnostic report for issue reporting. Use Refresh to update, then Copy to clipboard."),
        attr_name=dialog_attr_name,
    )
    # dlg.refresh_text() # redundant
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def create_generate_debug_log_action(
    *,
    parent: QWidget,
    recorder: InMemoryDebugRecorder | None = None,
    logger_name: str = "deeplabcut",
    libraries: Iterable[LibrarySpec | str] | None = None,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_module_paths: bool = False,
    include_executable_paths: bool = True,
    log_limit: int = 300,
    text: str = "&Generate debug log...",
    status_tip: str = "Generate a diagnostic report for troubleshooting",
    dialog_attr_name: str = "_dlc_debug_dialog",
) -> QAction:
    """
    Create a QAction that opens the DLC debug report dialog.

    Typical usage in ``MainWindow.create_actions``::

        self.generateDebugLogAction = create_generate_debug_log_action(parent=self)
    """
    action = QAction(text, parent)
    action.setStatusTip(status_tip)

    def _open_dialog() -> None:
        show_debug_report_dialog(
            parent=parent,
            recorder=recorder,
            logger_name=logger_name,
            libraries=libraries,
            executables=executables,
            include_module_paths=include_module_paths,
            include_executable_paths=include_executable_paths,
            log_limit=log_limit,
            dialog_attr_name=dialog_attr_name,
        )

    action.triggered.connect(_open_dialog)
    return action
