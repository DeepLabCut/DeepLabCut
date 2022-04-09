from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from widgets import ConfigEditor

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
    layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_vertical_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QVBoxLayout():

    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout

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
    def __init__(
        self, 
        button_label:str, 
        filetype:str = None , 
        cwd:str = None, 
        single_file:bool =False, 
        parent=None
        ):
        super(BrowseFilesButton, self).__init__(button_label)
        self.filetype = filetype
        self.single_file_only = single_file
        self.cwd = cwd
        self.parent = parent

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
        
        filepaths = open_file_func(
            self, 
            f"Select .{file_ext} files",
            cwd,
            f"Files (*.{file_ext})"
            )

        if filepaths:
            self.setProperty("files", filepaths[0])
        