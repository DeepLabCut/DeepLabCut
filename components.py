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
    lignment=None, spacing: int = 20, margins: tuple = (0, 0, 0, 0)
) -> QtWidgets.QGridLayout():

    layout = QtWidgets.QGridLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
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
        parent=None,
        dialog_text:str = None,
        file_text:str = None,
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
        
        filepaths = open_file_func(
            self, 
            dialog_text,
            cwd,
            file_text
            )

        if filepaths:
            # NOTE: how to store and access widget property
            self.setProperty("files", filepaths[0])
            print(self.Property("files"))
            