import logging
import os
from typing import List
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from dlc_params import DLC_Params

from widgets import ConfigEditor

from deeplabcut.utils import auxiliaryfunctions


def _create_label_widget(
    text: str, style: str = "", margins: tuple = (20, 50, 0, 10),
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
    lignment=None, spacing: int = 20, margins: tuple = None,
) -> QtWidgets.QGridLayout():

    layout = QtWidgets.QGridLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    if margins:
        layout.setContentsMargins(*margins)

    return layout


class BodypartListWidget(QtWidgets.QListWidget):
    def __init__(
        self, 
        root: QtWidgets.QMainWindow, 
        parent: QtWidgets.QWidget, 
        # all_bodyparts: List 
        # NOTE: Is there a case where a specific list should 
        # have bodyparts other than the root? I don't think so.
        ):
        super(BodypartListWidget, self).__init__()

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

    def update_selected_bodyparts(self):
        self.selected_bodyparts = [item.text() for item in self.selectedItems()]
        self.root.logger.info(
            f"Selected bodyparts:\n\t{self.selected_bodyparts}"
        )


class VideoSelectionWidget(QtWidgets.QWidget):
    # TODO: Selected should sync across tabs
    #       automatically! Probably need a slot
    #       in main window...
    def __init__(
        self, 
        root: QtWidgets.QMainWindow, 
        parent: QtWidgets.QWidget
    ):
        super(VideoSelectionWidget, self).__init__(parent)
        
        self.root = root
        self.parent = parent

        self.files = self.root.files
        
        self._init_layout()
    
    def _init_layout(self):
        layout = _create_horizontal_layout()

        # Videotype selection
        self.videotype_widget = QtWidgets.QComboBox()
        self.videotype_widget.setMaximumWidth(100)
        self.videotype_widget.addItems(DLC_Params.VIDEOTYPES)
        self.videotype_widget.setCurrentText(self.root.videotype)
        self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        # Select videos
        self.select_video_button = QtWidgets.QPushButton("Select videos")
        self.select_video_button.setMaximumWidth(200)
        self.select_video_button.clicked.connect(self.select_videos)

        # Number of selected videos text
        self.selected_videos_text = QtWidgets.QLabel("") #updated when videos are selected

        # Clear video selection
        self.clear_videos = QtWidgets.QPushButton("Clear selection")
        self.clear_videos.clicked.connect(self.clear_selected_videos)

        layout.addWidget(self.videotype_widget)
        layout.addWidget(self.select_video_button)
        layout.addWidget(self.selected_videos_text)
        layout.addWidget(self.clear_videos, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def update_videotype(self, vtype):
        self.clear_selected_videos()
        self.root.update_videotype(vtype)

    def _update_video_selection(self, videopaths):
        self.files.clear()
        self.files.update(
            videopaths
        )  
        if len(self.files)>0:
            self.selected_videos_text.setText("%s videos selected" % len(self.files))
            self.select_video_button.setText("Add more videos")
            # self.select_video_button.adjustSize()
        self.root.update_files(self.files)

    def select_videos(self):
        cwd = self.root.project_folder()
        filenames = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video(s) to analyze",
            cwd,
            f"Video files (*.{self.videotype_widget.currentText()})",
        )

        if filenames[0]:
            # Qt returns a tuple (list of files, filetype)
            self._update_video_selection(filenames[0]) 
            
    def clear_selected_videos(self):
        self.selected_videos_text.setText("")
        self.select_video_button.setText("Select videos")
        self.files.clear()
        self.root.files.clear()
        # self.select_video_button.adjustSize()
        self.root.logger.info(f"Cleared selected videos:\n{self.files}")


class TrainingSetSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(TrainingSetSpinBox, self).__init__(parent)
        
        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.trainingset_index)
        self.valueChanged.connect(self.root.update_trainingset)


class ShuffleSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(ShuffleSpinBox, self).__init__(parent)
        
        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.shuffle_value)
        self.valueChanged.connect(self.root.update_shuffle)


class DefaultTab(QtWidgets.QWidget):
    def __init__(
        self, 
        root: QtWidgets.QMainWindow, 
        parent: QtWidgets.QWidget, 
        h1_description: str
    ):
        super(DefaultTab, self).__init__(parent)

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
            _create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10),)
        )

        # Add separating line
        self.separatorLine = QtWidgets.QFrame()
        self.separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        self.separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        self.separatorLine.setLineWidth(0)
        self.separatorLine.setMidLineWidth(1)

        self.main_layout.addWidget(self.separatorLine)
        dummy_space = _create_label_widget("", margins=(0, 5, 0, 0))
        self.main_layout.addWidget(dummy_space)

        # Add config text field and button
        project_config_layout = _create_horizontal_layout()

        cfg_text = QtWidgets.QLabel("Active config file:")

        self.cfg_line = QtWidgets.QLineEdit()
        self.cfg_line.setText(self.root.config)
        self.cfg_line.textChanged[str].connect(self.root.update_cfg)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_cfg_file)

        project_config_layout.addWidget(cfg_text)
        project_config_layout.addWidget(self.cfg_line)
        project_config_layout.addWidget(browse_button)

        self.main_layout.addLayout(project_config_layout)

    def browse_cfg_file(self):
        cwd = self.root.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config[0]:
            warning_dialog = QtWidgets.QMessageBox()
            warning_dialog.setWindowTitle("Warning!")
            warning_dialog.setIcon(QtWidgets.QMessageBox.Warning)
            warning_dialog.setText("No config file selected...")
            button = warning_dialog.exec_()
            return

        self.root.config = config[0]
        self.root.logger.info(f"Changed config file: {self.root.config}")
        self.cfg_line.setText(self.root.config)


class EditYamlButton(QtWidgets.QPushButton):
    def __init__(
        self, 
        button_label: str, 
        filepath: str, 
        parent: QtWidgets.QWidget = None
    ):
        super(EditYamlButton, self).__init__(button_label)
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
        super(BrowseFilesButton, self).__init__(button_label)
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
