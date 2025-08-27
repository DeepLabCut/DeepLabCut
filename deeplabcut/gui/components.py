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

import os

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.widgets import ConfigEditor
from pathlib import Path


def _create_label_widget(
    text: str,
    style: str = "",
    margins: tuple = (20, 10, 0, 10),
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setContentsMargins(*margins)
    label.setStyleSheet(style)

    return label


def _create_horizontal_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QHBoxLayout():
    layout = QtWidgets.QHBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_vertical_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QVBoxLayout():
    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_grid_layout(
    alignment=None,
    spacing: int = 20,
    margins: tuple = None,
) -> QtWidgets.QGridLayout:
    layout = QtWidgets.QGridLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    if margins:
        layout.setContentsMargins(*margins)

    return layout


def set_combo_items(combo_box: QtWidgets.QComboBox, items: list[str], index: int = 0):
    """
    Safely replaces all items in a QComboBox and sets the current index,
    ensuring that the `currentTextChanged` signal is emitted exactly once
    (and only if items are present).

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
    combo_box.blockSignals(True)
    combo_box.clear()
    combo_box.addItems(items)
    combo_box.blockSignals(False)

    if not items:
        combo_box.setCurrentIndex(-1)
        return

    current = combo_box.currentIndex()
    if current == index:
        # Temporarily change index to suppress duplicate signal
        combo_box.blockSignals(True)
        combo_box.setCurrentIndex(-1)
        combo_box.blockSignals(False)

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
    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget):
        super().__init__(parent)

        self.root = root
        self.parent = parent

        self._init_layout()

    def _init_layout(self):
        layout = _create_horizontal_layout()

        # Videotype selection
        self.videotype_widget = QtWidgets.QComboBox()
        self.videotype_widget.setMinimumWidth(100)
        self.videotype_widget.addItems(DLCParams.VIDEOTYPES)
        self.videotype_widget.setCurrentText(self.root.video_type)
        self.root.video_type_.connect(self.videotype_widget.setCurrentText)
        self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        # Select videos
        self.select_video_button = QtWidgets.QPushButton("Select videos")
        self.select_video_button.setMaximumWidth(200)
        self.select_video_button.clicked.connect(self.update_videos)
        self.root.video_files_.connect(self._update_video_selection)

        # Number of selected videos text
        self.selected_videos_text = QtWidgets.QLabel(
            ""
        )  # updated when videos are selected

        # Clear video selection
        self.clear_videos = QtWidgets.QPushButton("Clear selection")
        self.clear_videos.clicked.connect(self.clear_selected_videos)

        layout.addWidget(self.videotype_widget)
        layout.addWidget(self.select_video_button)
        layout.addWidget(self.selected_videos_text)
        layout.addWidget(self.clear_videos, alignment=Qt.AlignRight)

        self.setLayout(layout)

    @property
    def files(self):
        return self.root.video_files

    def update_videotype(self, vtype):
        self.clear_selected_videos()
        self.root.video_type = vtype

    def _update_video_selection(self, videopaths):
        n_videos = len(self.root.video_files)
        if n_videos:
            self.selected_videos_text.setText(f"{n_videos} videos selected")
            self.select_video_button.setText("Add more videos")
        else:
            self.selected_videos_text.setText("")
            self.select_video_button.setText("Select videos")

    def update_videos(self):
        directory_to_open = self.root.project_folder

        # Create a filter string with both lowercase and uppercase extensions

        video_types = [f"*.{ext.lower()}" for ext in DLCParams.VIDEOTYPES[1:]] + [
            f"*.{ext.upper()}" for ext in DLCParams.VIDEOTYPES[1:]
        ]
        video_filter = f"Videos ({' '.join(video_types)})"

        filenames = QtWidgets.QFileDialog.getOpenFileNames(
            parent=self,
            caption="Select video(s) to analyze",
            dir=directory_to_open,
            filter=video_filter,
        )

        if filenames[0]:
            # Qt returns a tuple (list of files, filetype)
            self.root.add_video_files([os.path.abspath(vid) for vid in filenames[0]])

    def clear_selected_videos(self):
        self.root.clear_video_files()
        self.root.logger.info(f"Cleared selected videos")


class MediaSelectionWidget(QtWidgets.QWidget):
    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget, hide_videotype: bool = False):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.hide_videotype = hide_videotype
        self._init_layout()

    def _init_layout(self):
        layout = _create_horizontal_layout()

        # Media type selection
        self.media_type_widget = QtWidgets.QComboBox()
        self.media_type_widget.setMinimumWidth(100)
        self.media_type_widget.addItems(["Videos", "Images"])
        self.media_type_widget.currentTextChanged.connect(self.update_media_type)

        # Videotype selection (for videos)
        self.videotype_widget = QtWidgets.QComboBox()
        self.videotype_widget.setMinimumWidth(100)
        self.videotype_widget.addItems(DLCParams.VIDEOTYPES)
        self.videotype_widget.setCurrentText(self.root.video_type)
        self.root.video_type_.connect(self.videotype_widget.setCurrentText)
        self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        # Select media button
        self.select_media_button = QtWidgets.QPushButton("Select files")
        self.select_media_button.setMaximumWidth(200)
        self.select_media_button.clicked.connect(self.update_media)
        self.root.video_files_.connect(self._update_media_selection)

        # Number of selected files text
        self.selected_files_text = QtWidgets.QLabel(
            ""
        )  # updated when files are selected

        # Clear selection
        self.clear_files = QtWidgets.QPushButton("Clear selection")
        self.clear_files.clicked.connect(self.clear_selected_files)

        layout.addWidget(self.media_type_widget)
        if not self.hide_videotype:
            layout.addWidget(self.videotype_widget)
        layout.addWidget(self.select_media_button)
        layout.addWidget(self.selected_files_text)
        layout.addWidget(self.clear_files, alignment=Qt.AlignRight)

        self.setLayout(layout)

    @property
    def files(self):
        return self.root.video_files

    def update_media_type(self, media_type):
        self.clear_selected_files()
        if media_type == "Videos" and not self.hide_videotype:
            self.videotype_widget.show()
        elif not self.hide_videotype:
            self.videotype_widget.hide()

    def update_videotype(self, vtype):
        self.clear_selected_files()
        self.root.video_type = vtype

    def _update_media_selection(self, filepaths):
        n_files = len(self.root.video_files)
        if n_files:
            media_type = self.media_type_widget.currentText().lower()
            self.selected_files_text.setText(f"{n_files} {media_type} selected")
            self.select_media_button.setText(f"Add more {media_type}")
        else:
            self.selected_files_text.setText("")
            self.select_media_button.setText("Select files")

    def update_media(self):
        directory_to_open = self.root.project_folder
        media_type = self.media_type_widget.currentText()

        if media_type == "Videos":
            # Create a filter string with both lowercase and uppercase extensions
            video_types = [f"*.{ext.lower()}" for ext in DLCParams.VIDEOTYPES[1:]] + [
                f"*.{ext.upper()}" for ext in DLCParams.VIDEOTYPES[1:]
            ]
            media_filter = f"Videos ({' '.join(video_types)})"
            caption = "Select video(s) to analyze"
        else:  # Images
            image_types = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
            media_filter = f"Images ({' '.join(image_types)})"
            caption = "Select image(s) to analyze"

        filenames = QtWidgets.QFileDialog.getOpenFileNames(
            parent=self,
            caption=caption,
            dir=directory_to_open,
            filter=media_filter,
        )

        if filenames[0]:
            # Qt returns a tuple (list of files, filetype)
            self.root.add_video_files([os.path.abspath(f) for f in filenames[0]])

    def clear_selected_files(self):
        self.root.clear_video_files()
        self.root.logger.info(f"Cleared selected files")


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
        self.selected_snapshot = None
        self._init_layout(margins, select_button_text)

    def _init_layout(self, margins, select_button_text):
        layout = _create_horizontal_layout(margins=margins)

        # Select snapshot
        self.select_snapshot_button = QtWidgets.QPushButton(select_button_text)
        self.select_snapshot_button.setMaximumWidth(200)
        self.select_snapshot_button.clicked.connect(self.select_snapshot)

        # Selected snapshot text
        self.selected_snapshot_text = QtWidgets.QLabel(
            ""
        )  # updated when snapshot is selected

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
            self.selected_snapshot_text.setText(
                f"{os.path.basename(self.selected_snapshot)}"
            )
            self.clear_snapshot_button.show()

    def select_snapshot(self):
        # Create a filter string with both lowercase and uppercase extensions
        snapshot_types = ["*.pt", "*.PT"]
        snapshot_filter = f"Snapshots ({' '.join(snapshot_types)})"

        directory_to_open = self.root.models_folder

        selected_snapshot, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select snapshot to start training from",
            dir=directory_to_open,
            filter=snapshot_filter,
        )
        # When Canceling a file selection, Qt returns an empty string as selected file
        if selected_snapshot:
            self.selected_snapshot = os.path.abspath(selected_snapshot)

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
        self.selected_conditions = None
        self._init_layout()

    def _init_layout(self):
        layout = _create_horizontal_layout()

        # Select conditions
        self.select_conditions_button = QtWidgets.QPushButton("Select conditions")
        self.select_conditions_button.setMaximumWidth(200)
        self.select_conditions_button.clicked.connect(self.select_conditions)

        # Selected conditions text
        self.selected_conditions_text = QtWidgets.QLabel(
            ""
        )  # updated when conditions are selected

        layout.addWidget(self.select_conditions_button)
        layout.addWidget(self.selected_conditions_text)

        self.setLayout(layout)

    def _update_selected_conditions_display(self):
        def _shorten_path(path: str, max_length: int = 30) -> str:
            if len(path) <= max_length:
                return path
            return "..." + path[-(max_length - 3) :]

        self.selected_conditions_text.setText(
            ""
            if self.selected_conditions is None
            else f"{_shorten_path(self.selected_conditions)}"
        )

    def select_conditions(self):
        def _is_model_bu(selected_conditions) -> bool:
            model_config_path = Path(selected_conditions).parent / "pytorch_config.yaml"
            model_config = read_config_as_dict(model_config_path)
            return model_config.get("method").lower() == "bu"

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

        directory_to_open = self.root.project_folder

        selected_conditions, selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select conditions to use during inference (snapshot or predictions file)",
            dir=directory_to_open,
            filter=conditions_filter,
        )
        if selected_filter.startswith(snapshots_label) and selected_conditions:
            if not _is_model_bu(selected_conditions):
                msg = _create_message_box(
                    f"Invalid conditions",
                    (
                        f"The selected snapshot ({selected_conditions}) cannot be "
                        "used as conditions because it is not a Bottom-Up model."
                    ),
                )
                msg.exec_()
                selected_conditions = None

        # When Canceling a file selection, Qt returns an empty string as selected file
        self.selected_conditions = (
            str(os.path.abspath(selected_conditions)) if selected_conditions else None
        )

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
    def update_shuffle(self, new_shuffle: int):
        if new_shuffle != self.value():
            self.setValue(new_shuffle)


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
        self.main_layout.addWidget(
            _create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10))
        )

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
    def __init__(
        self, button_label: str, filepath: str, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.filepath = filepath
        self.parent = parent

        self.clicked.connect(self.open_config)

    def open_config(self):
        editor = ConfigEditor(self.filepath)
        editor.show()


class BrowseFilesButton(QtWidgets.QPushButton):
    def __init__(
        self,
        button_label: str,
        filetype: str = None,
        cwd: str = None,
        single_file: bool = False,
        dialog_text: str = None,
        file_text: str = None,
        parent=None,
    ):
        super().__init__(parent)
        self.filetype = filetype
        self.single_file_only = single_file
        self.cwd = cwd
        self.parent = parent

        self.dialog_text = dialog_text
        self.file_text = file_text

        self.files = set()

        self.clicked.connect(self.browse_files)

    def browse_files(self):
        # Look for any extension by default
        file_ext = "*"
        if self.filetype:
            # This works both with e.g. .avi and avi
            file_ext = self.filetype.split(".")[-1]

        # Choose multiple files by default
        open_file_func = QtWidgets.QFileDialog.getOpenFileNames
        if self.single_file_only:
            open_file_func = QtWidgets.QFileDialog.getOpenFileName

        cwd = ""
        if self.cwd:
            cwd = self.cwd

        dialog_text = f"Select .{file_ext} files"
        if self.dialog_text:
            dialog_text = self.dialog_text

        file_text = f"Files (*.{file_ext})"
        if self.file_text:
            file_text = self.file_text

        filepaths = open_file_func(self, dialog_text, cwd, file_text)

        if filepaths:
            self.files.update(filepaths[0])


def _create_message_box(text, info_text):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info_text)

    msg.setWindowTitle("Info")
    msg.setMinimumWidth(900)
    logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg


def _create_confirmation_box(title, description):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(title)
    msg.setInformativeText(description)

    msg.setWindowTitle("Confirmation")
    msg.setMinimumWidth(900)
    logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    return msg
