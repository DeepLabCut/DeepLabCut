import logging
import os
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from widgets import ConfigEditor

from deeplabcut.utils import auxiliaryfunctions


def _create_label_widget(
    text: str, style: str = "", margins: tuple = (20, 50, 0, 0),
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


class DefaultTab(QtWidgets.QWidget):
    def __init__(self, parent, tab_heading):
        super(DefaultTab, self).__init__(parent)

        self.logger = logging.getLogger("GUI")

        self.parent = parent
        self.config = self.parent.config
        self.cfg = self.parent.cfg

        self.tab_heading = tab_heading

        # NOTE: Does it make sense to add other commong info?
        # Like shuffle, trainingsetidx

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.main_layout)

        self._init_default_layout()

        self.logger.debug("Initializing default tab window...")
        self.logger.debug(f"Config file: {self.config}")
        self.logger.debug(f"Config dict: {self.cfg}")
        self.logger.debug(f"Project folder: {self.project_folder}")
        self.logger.debug(f"Is multianimal: {self.is_multianimal}")
        self.logger.debug(f"All bodyparts: {self.all_bodyparts}")

    @property
    def project_folder(self):
        return self.cfg.get("project_path", os.path.expanduser("~/Desktop"))

    @property
    def is_multianimal(self):
        if self.cfg["multianimalproject"]:
            return True
        else:
            return False

    @property
    def all_bodyparts(self):
        if self.is_multianimal:
            return self.cfg["multianimalbodyparts"]
        else:
            return self.cfg["bodyparts"]

    def _init_default_layout(self):
        # Add tab header
        self.main_layout.addWidget(
            _create_label_widget(self.tab_heading, "font:bold;", (10, 10, 0, 10),)
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
        self.cfg_line.setMinimumHeight(30)
        self.cfg_line.setText(self.config)
        self.cfg_line.textChanged[str].connect(self.update_cfg)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.setMinimumHeight(30)
        browse_button.clicked.connect(self.browse_cfg_file)

        project_config_layout.addWidget(cfg_text)
        project_config_layout.addWidget(self.cfg_line)
        project_config_layout.addWidget(browse_button)

        self.main_layout.addLayout(project_config_layout)

    def browse_cfg_file(self):
        cwd = self.config
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

        self.config = config[0]
        self.logger.info(f"Changed config file: {self.config}")
        self.cfg_line.setText(self.config)

    def update_cfg(self):
        text = self.cfg_line.text()
        self.config = text
        self.cfg = auxiliaryfunctions.read_config(self.config)


class EditYamlButton(QtWidgets.QPushButton):
    def __init__(self, button_label, filepath, parent=None):
        super(EditYamlButton, self).__init__(button_label)
        self.filepath = filepath
        self.parent = parent

        self.clicked.connect(self.open_config)

    def open_config(self):
        editor = ConfigEditor(self.filepath)
        editor.show()


class BrowseFilesButton(QtWidgets.QPushButton):
    # NOTE: This is not functioning as intended yet. I dont know how
    #       to store and retrieve information in the button, so that it
    #       can be accessed elsewhere.
    def __init__(
        self,
        button_label: str,
        filetype: str = None,
        cwd: str = None,
        single_file: bool = False,
        parent=None,
        dialog_text: str = None,
        file_text: str = None,
    ):
        super(BrowseFilesButton, self).__init__(button_label)
        self.filetype = filetype
        self.single_file_only = single_file
        self.cwd = cwd
        self.parent = parent

        self.dialog_text = dialog_text
        self.file_text = file_text
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
            # NOTE: how to store and access widget property
            self.setProperty("files", filepaths[0])
            print(self.Property("files"))

