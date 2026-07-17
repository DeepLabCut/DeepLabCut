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
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.gui_assets import icon_from_resource
from deeplabcut.gui.widgets import ConfigEditor

PathInput = str | Path
Margins = tuple[int, int, int, int]


def _create_label_widget(
    text: str,
    style: str = "",
    margins: Margins = (20, 10, 0, 10),
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setContentsMargins(*margins)
    label.setStyleSheet(style)
    return label


def _create_horizontal_layout(
    alignment: Qt.AlignmentFlag | None = None,
    spacing: int = 20,
    margins: Margins = (20, 0, 0, 0),
) -> QtWidgets.QHBoxLayout:
    layout = QtWidgets.QHBoxLayout()
    layout.setAlignment(alignment if alignment is not None else Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


def _create_vertical_layout(
    alignment: Qt.AlignmentFlag | None = None,
    spacing: int = 20,
    margins: Margins = (20, 0, 0, 0),
) -> QtWidgets.QVBoxLayout:
    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(alignment if alignment is not None else Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


def _create_grid_layout(
    alignment: Qt.AlignmentFlag | None = None,
    spacing: int = 20,
    margins: Margins | None = None,
) -> QtWidgets.QGridLayout:
    layout = QtWidgets.QGridLayout()
    layout.setAlignment(alignment if alignment is not None else Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)

    if margins is not None:
        layout.setContentsMargins(*margins)

    return layout


def _dialog_directory(directory: PathInput | None) -> str:
    """Convert an optional path to a QFileDialog-compatible string."""
    if directory is None or directory == "":
        return ""
    return str(directory)


def _get_open_file_name(
    parent: QtWidgets.QWidget,
    caption: str,
    directory: PathInput | None,
    file_filter: str,
) -> tuple[str, str]:
    """Open a binding-compatible single-file dialog."""
    return QtWidgets.QFileDialog.getOpenFileName(
        parent,
        caption,
        _dialog_directory(directory),
        file_filter,
    )


def _get_open_file_names(
    parent: QtWidgets.QWidget,
    caption: str,
    directory: PathInput | None,
    file_filter: str,
) -> tuple[list[str], str]:
    """Open a binding-compatible multiple-file dialog."""
    return QtWidgets.QFileDialog.getOpenFileNames(
        parent,
        caption,
        _dialog_directory(directory),
        file_filter,
    )


def set_combo_items(combo_box: QtWidgets.QComboBox, items: list[str], index: int = 0) -> None:
    """Safely replaces all items in a QComboBox and sets the current index, ensuring
    that the `currentTextChanged` signal is emitted exactly once (and only if items are
    present).

    This method suppresses intermediate signal emissions that can be triggered
    by `clear()` and `addItems()` — both of which may emit multiple signals
    depending on the underlying Qt model and signal connections.

    It also handles the edge case where the item at the target index is already
    selected: by default, Qt will not emit a signal if the index doesn't change.
    To ensure consistent behavior, this method temporarily sets the index to -1
    (i.e., no selection), which is done with signals blocked, then restores the
    intended index — causing the signal to emit once and only once.

    Parameters:
        combo_box (QComboBox): The combo box to update.
        items (list of str): New items to populate the combo box.
        index (int): The index to select after updating items. Defaults to 0.

    Note:
        - If the items list is empty, no item will be selected and no signal will be emitted.
        - This method is designed to be safe for use with PySide, where signals
          cannot be manually emitted, and future-proof if multiple slots are connected.
    """

    previous = combo_box.blockSignals(True)
    try:
        combo_box.clear()
        combo_box.addItems(items)

        if not items:
            combo_box.setCurrentIndex(-1)
            return

        if combo_box.currentIndex() == index:
            combo_box.setCurrentIndex(-1)
    finally:
        combo_box.blockSignals(previous)

    combo_box.setCurrentIndex(index)


class BodypartListWidget(QtWidgets.QListWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget,
        # all_bodyparts: List
        # NOTE: Is there a case where a specific list should
        # have bodyparts other than the root? I don't think so.
    ):
        super().__init__()

        self.root = root
        self.parent = parent
        self.selected_bodyparts = self.root.all_bodyparts

        self.setEnabled(False)
        self.setMaximumWidth(600)
        self.setMaximumHeight(500)
        self.hide()

        self.addItems(self.root.all_bodyparts)
        self.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.itemSelectionChanged.connect(self.update_selected_bodyparts)

    def refresh(self):
        self.clear()
        self.addItems(self.root.all_bodyparts)
        self.update_selected_bodyparts()

    def update_selected_bodyparts(self):
        self.selected_bodyparts = [item.text() for item in self.selectedItems()]
        self.root.logger.info(f"Selected bodyparts:\n\t{self.selected_bodyparts}")


class VideoSelectionWidget(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget,
        *,
        hide_videotype: bool = False,
        sync_videotype_with_selection: bool = False,
        strict_videotype_filter: bool = False,
    ):
        super().__init__(parent)

        self.root = root
        self.parent = parent

        # Optional safeties; defaults preserve current behavior
        self.sync_videotype_with_selection = sync_videotype_with_selection
        self.strict_videotype_filter = strict_videotype_filter

        self._init_layout(hide_videotype)

    def _init_layout(self, hide_videotype: bool):
        layout = _create_horizontal_layout()

        # Videotype selection
        self.videotype_widget = QtWidgets.QComboBox()
        self.videotype_widget.setMinimumWidth(100)
        self.videotype_widget.addItems(DLCParams.VIDEOTYPES)
        self.videotype_widget.setCurrentText(self._normalize_videotype(self.root.video_type))
        self.root.video_type_.connect(self._sync_videotype_from_root)
        self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        # Select videos
        self.select_video_button = QtWidgets.QPushButton("Select videos")
        self.select_video_button.setMaximumWidth(200)
        self.select_video_button.clicked.connect(self.update_videos)
        self.root.video_files_.connect(self._update_video_selection)

        # Number of selected videos text
        self.selected_videos_text = QtWidgets.QLabel("")

        # Clear video selection
        self.clear_videos = QtWidgets.QPushButton("Clear selection")
        self.clear_videos.clicked.connect(self.clear_selected_videos)

        if not hide_videotype:
            layout.addWidget(self.videotype_widget)
        layout.addWidget(self.select_video_button)
        layout.addWidget(self.selected_videos_text)
        layout.addWidget(self.clear_videos, alignment=Qt.AlignRight)

        self.setLayout(layout)

    @property
    def files(self):
        return self.root.video_files

    def _normalize_videotype(self, vtype: str) -> str:
        return (vtype or "").lower().lstrip(".")

    @property
    def selected_suffixes(self) -> set[str]:
        """Return normalized suffixes (without leading dot) of currently selected files."""
        return {Path(video).suffix.lower().lstrip(".") for video in self.files if Path(video).suffix}

    @Slot(str)
    def _sync_videotype_from_root(self, vtype: str) -> None:
        normalized = self._normalize_videotype(vtype)

        if normalized == self._normalize_videotype(self.videotype_widget.currentText()):
            return

        previous = self.videotype_widget.blockSignals(True)
        try:
            self.videotype_widget.setCurrentText(normalized)
        finally:
            self.videotype_widget.blockSignals(previous)

    def get_effective_videotype(
        self,
        prefer_selected_files: bool = False,
        with_dot: bool = True,
    ) -> str:
        """
        Return the videotype to use.

        By default, preserves current behavior and uses the dropdown.
        If prefer_selected_files=True and the selected files all share one suffix,
        that suffix is used instead.
        """
        videotype = self._normalize_videotype(self.videotype_widget.currentText())

        if prefer_selected_files:
            suffixes = self.selected_suffixes
            if len(suffixes) == 1:
                videotype = next(iter(suffixes))

        if with_dot and videotype:
            return f".{videotype}"
        return videotype

    def get_files_grouped_by_suffix(
        self,
        keep_dot: bool = False,
    ) -> dict[str, list[Path]]:
        """Return selected files grouped by suffix."""
        groups: dict[str, list[Path]] = {}

        for video in self.files:
            path = Path(video)
            suffix = path.suffix.lower()

            if not keep_dot:
                suffix = suffix.lstrip(".")

            groups.setdefault(suffix, []).append(path)

        return groups

    def _all_supported_video_patterns(self) -> list[str]:
        """Return all supported video patterns in both lower and upper case."""
        return [f"*.{ext.lower()}" for ext in DLCParams.VIDEOTYPES[1:]] + [
            f"*.{ext.upper()}" for ext in DLCParams.VIDEOTYPES[1:]
        ]

    def _build_video_filter(self) -> str:
        """
        Build the file dialog filter.

        By default, preserve current behavior: show all supported video types.
        If strict_videotype_filter is enabled, restrict to the currently selected
        videotype when it is non-empty. If the current dropdown value is empty
        (the "all types" option), fall back to the full supported-extension filter.
        """
        all_video_types = self._all_supported_video_patterns()

        if self.strict_videotype_filter:
            current = self.get_effective_videotype(
                prefer_selected_files=False,
                with_dot=False,
            )

            if current:
                video_types = [f"*.{current.lower()}", f"*.{current.upper()}"]
            else:
                # "All types" entry selected: keep the dialog usable
                video_types = all_video_types
        else:
            video_types = all_video_types

        return f"Videos ({' '.join(video_types)})"

    def _set_videotype_silently(self, vtype: str) -> None:
        normalized = self._normalize_videotype(vtype)
        current = self._normalize_videotype(self.videotype_widget.currentText())

        if not normalized:
            self.root.logger.warning("Attempted to set an empty videotype silently; keeping current selection.")
            return

        if self.videotype_widget.findText(normalized) == -1:
            self.root.logger.warning(
                f"Attempted to set unsupported videotype "
                f"{normalized!r} silently; keeping current videotype "
                f"{current!r}."
            )
            return

        previous = self.videotype_widget.blockSignals(True)
        try:
            self.videotype_widget.setCurrentText(normalized)
        finally:
            self.videotype_widget.blockSignals(previous)

        if self._normalize_videotype(self.root.video_type) != normalized:
            self.root.video_type = normalized

    @Slot(str)
    def update_videotype(self, vtype: str) -> None:
        normalized = self._normalize_videotype(vtype)
        current = self._normalize_videotype(self.root.video_type)

        if normalized == current:
            return

        self.clear_selected_videos()
        self.root.video_type = normalized

    def _update_video_selection(self, _videopaths) -> None:
        n_videos = len(self.root.video_files)

        if not n_videos:
            self.selected_videos_text.setText("")
            self.select_video_button.setText("Select videos")
            return

        suffixes = self.selected_suffixes

        if len(suffixes) == 1:
            suffix = next(iter(suffixes))
            text = f"{n_videos} videos selected (.{suffix})"
        elif len(suffixes) > 1:
            counts = {suffix: len(files) for suffix, files in self.get_files_grouped_by_suffix().items()}
            summary = ", ".join(f"{count} .{suffix}" for suffix, count in sorted(counts.items()))
            text = f"{n_videos} videos selected ({summary}; will run in separate batches)"
        else:
            text = f"{n_videos} videos selected"

        self.selected_videos_text.setText(text)
        self.select_video_button.setText("Add more videos")

    def update_videos(self):
        video_filter = self._build_video_filter()

        filenames, _ = _get_open_file_names(
            self,
            "Select video(s) to analyze",
            self.root.project_folder,
            video_filter,
        )

        if not filenames:
            return

        abs_files = [Path(filename).absolute() for filename in filenames]
        self.root.add_video_files(abs_files)

        if not self.sync_videotype_with_selection:
            return

        suffixes = {video.suffix.lower().lstrip(".") for video in abs_files if video.suffix}

        if len(suffixes) == 1:
            inferred = next(iter(suffixes))
            self._set_videotype_silently(inferred)
            self.root.logger.info(f"Inferred videotype {inferred!r} from selected file(s)")
        elif len(suffixes) > 1:
            self.root.logger.warning(
                f"Selected videos have mixed suffixes {sorted(suffixes)}; keeping current videotype dropdown unchanged."
            )

    def clear_selected_videos(self):
        self.root.clear_video_files()
        self.root.logger.debug("Cleared selected videos")


class SnapshotSelectionWidget(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget,
        margins: tuple,
        select_button_text: str,
    ):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.selected_snapshot: Path | None = None
        self._init_layout(margins, select_button_text)

    def _init_layout(self, margins, select_button_text):
        layout = _create_horizontal_layout(margins=margins)

        # Select snapshot
        self.select_snapshot_button = QtWidgets.QPushButton(select_button_text)
        self.select_snapshot_button.setMaximumWidth(200)
        self.select_snapshot_button.clicked.connect(self.select_snapshot)

        # Selected snapshot text
        self.selected_snapshot_text = QtWidgets.QLabel("")  # updated when snapshot is selected

        # Clear snapshot selection
        self.clear_snapshot_button = QtWidgets.QPushButton("Clear selection")
        self.clear_snapshot_button.clicked.connect(self.clear_selected_snapshot)
        self.clear_snapshot_button.hide()

        layout.addWidget(self.select_snapshot_button)
        layout.addWidget(self.selected_snapshot_text)
        layout.addWidget(self.clear_snapshot_button, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def _update_selected_snapshot_display(self):
        if self.selected_snapshot is None:
            self.selected_snapshot_text.setText("")
            self.clear_snapshot_button.hide()
        else:
            self.selected_snapshot_text.setText(self.selected_snapshot.name)
            self.clear_snapshot_button.show()

    def select_snapshot(self):
        snapshot_types = ["*.pt", "*.PT"]
        snapshot_filter = f"Snapshots ({' '.join(snapshot_types)})"

        selected_snapshot, _ = _get_open_file_name(
            self,
            "Select snapshot to start training from",
            self.root.models_folder,
            snapshot_filter,
        )

        if selected_snapshot:
            self.selected_snapshot = Path(selected_snapshot).absolute()

        self._update_selected_snapshot_display()

    def clear_selected_snapshot(self):
        self.selected_snapshot = None
        self._update_selected_snapshot_display()


class ConditionsSelectionWidget(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget,
    ):
        super().__init__(parent=parent)
        self.root = root
        self.parent = parent
        self.selected_conditions: Path | None = None
        self._init_layout()

    def _init_layout(self):
        layout = _create_horizontal_layout()

        # Select conditions
        self.select_conditions_button = QtWidgets.QPushButton("Select conditions")
        self.select_conditions_button.setMaximumWidth(200)
        self.select_conditions_button.clicked.connect(self.select_conditions)

        # Selected conditions text
        self.selected_conditions_text = QtWidgets.QLabel("")  # updated when conditions are selected

        layout.addWidget(self.select_conditions_button)
        layout.addWidget(self.selected_conditions_text)

        self.setLayout(layout)

    def _update_selected_conditions_display(self):
        def _shorten_path(path: Path | str, max_length: int = 30) -> str:
            path_str = str(path)
            if len(path_str) <= max_length:
                return path_str
            return "..." + path_str[-(max_length - 3) :]

        self.selected_conditions_text.setText(
            "" if self.selected_conditions is None else _shorten_path(self.selected_conditions)
        )

    def select_conditions(self):
        def _is_model_bu(
            selected_conditions: PathInput,
        ) -> bool:
            model_config_path = Path(selected_conditions).parent / "pytorch_config.yaml"
            model_config = read_config_as_dict(model_config_path)
            method = model_config.get("method")

            return isinstance(method, str) and method.lower() == "bu"

        # Create a filter string with both lowercase and uppercase extensions
        snapshots_label = "Snapshots"
        h5_predictions_label = "H5 predictions"
        json_prediction_label = "Json predictions"
        snapshot_types = ["*.pt", "*.PT"]
        h5_predictions_types = ["*.h5", "*.H5"]
        json_prediction_types = ["*.json", "*.JSON"]
        conditions_filter = ";;".join(
            [
                f"{snapshots_label} ({' '.join(snapshot_types)})",
                f"{h5_predictions_label} ({' '.join(h5_predictions_types)})",
                f"{json_prediction_label} ({' '.join(json_prediction_types)})",
            ]
        )

        selected_conditions, selected_filter = _get_open_file_name(
            self,
            ("Select conditions to use during inference (snapshot or predictions file)"),
            self.root.project_folder,
            conditions_filter,
        )

        if (
            selected_conditions
            and selected_filter.startswith(snapshots_label)
            and not _is_model_bu(selected_conditions)
        ):
            msg = _create_message_box(
                "Invalid conditions",
                (
                    f"The selected snapshot "
                    f"({selected_conditions}) cannot be used "
                    "as conditions because it is not a "
                    "Bottom-Up model."
                ),
            )
            msg.exec()
            selected_conditions = None

        self.selected_conditions = Path(selected_conditions).absolute() if selected_conditions else None

        self._update_selected_conditions_display()


class TrainingSetSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super().__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.trainingset_index)
        self.valueChanged.connect(self.root.update_trainingset)


class ShuffleSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super().__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(10_000)
        self.setValue(self.root.shuffle_value)
        self.valueChanged.connect(self.root.update_shuffle)
        self.root.shuffle_change.connect(self.update_shuffle)

    @Slot(int)
    def update_shuffle(self, new_shuffle: int) -> None:
        if new_shuffle == self.value():
            return

        previous = self.blockSignals(True)
        try:
            self.setValue(new_shuffle)
        finally:
            self.blockSignals(previous)


class DefaultTab(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget = None,
        h1_description: str = "",
    ):
        super().__init__(parent)

        self.parent = parent
        self.root = root

        self.h1_description = h1_description

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self.main_layout)

        self._init_default_layout()

    def _init_default_layout(self):
        # Add tab header
        self.main_layout.addWidget(_create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10)))

        # Add separating line
        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Raised)
        self.separator.setLineWidth(0)
        self.separator.setMidLineWidth(1)
        policy = QtWidgets.QSizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.separator.setSizePolicy(policy)
        self.main_layout.addWidget(self.separator)


class EditYamlButton(QtWidgets.QPushButton):
    def __init__(self, button_label: str, filepath: str, parent: QtWidgets.QWidget = None):
        super().__init__(button_label, parent)
        self.filepath = filepath
        self.parent = parent
        self._editor: ConfigEditor | None = None

        self.clicked.connect(self.open_config)

    def open_config(self):
        self._editor = ConfigEditor(self.filepath)
        self._editor.show()


class BrowseFilesButton(QtWidgets.QPushButton):
    def __init__(
        self,
        button_label: str,
        filetype: str = None,
        cwd: PathInput | None = None,
        single_file: bool = False,
        dialog_text: str = None,
        file_text: str = None,
        parent=None,
    ):
        super().__init__(button_label, parent)
        self.filetype = filetype
        self.single_file_only = single_file
        self.cwd = cwd
        self.parent = parent

        self.dialog_text = dialog_text
        self.file_text = file_text

        self.files: set[Path] = set()

        self.clicked.connect(self.browse_files)

    def browse_files(self) -> None:
        file_ext = "*"
        if self.filetype:
            file_ext = self.filetype.rsplit(".", 1)[-1]

        dialog_text = self.dialog_text or f"Select .{file_ext} files"
        file_text = self.file_text or f"Files (*.{file_ext})"

        if self.single_file_only:
            filepath, _ = _get_open_file_name(
                self,
                dialog_text,
                self.cwd,
                file_text,
            )
            if filepath:
                self.files.add(Path(filepath).absolute())
            return

        filepaths, _ = _get_open_file_names(
            self,
            dialog_text,
            self.cwd,
            file_text,
        )
        self.files.update(Path(filepath).absolute() for filepath in filepaths)


def _create_message_box(text, info_text):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info_text)

    msg.setWindowTitle("Info")
    msg.setMinimumWidth(900)
    icon = icon_from_resource("logo.png")
    msg.setWindowIcon(icon)
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg


def _create_confirmation_box(title, description):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(title)
    msg.setInformativeText(description)

    msg.setWindowTitle("Confirmation")
    msg.setMinimumWidth(900)
    icon = icon_from_resource("logo.png")
    msg.setWindowIcon(icon)
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    return msg


def set_layout_contents_visible(layout: QtWidgets.QLayout, visible: bool):
    for i in range(layout.count()):
        item = layout.itemAt(i)

        # If it's a widget item
        widget = item.widget()
        if widget is not None:
            widget.setVisible(visible)

        # If it's a nested layout
        child_layout = item.layout()
        if child_layout is not None:
            set_layout_contents_visible(child_layout, visible)
