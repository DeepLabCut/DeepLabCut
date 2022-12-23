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
import os

from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QCheckBox


class OpenProject(QtWidgets.QDialog):
    def __init__(self, parent):
        super(OpenProject, self).__init__(parent)

        self.setWindowTitle("Load Existing Project")

        self.config = None
        self.loaded = False

        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_open()

        self.ok_button = QtWidgets.QPushButton("Load")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.open_project)

        main_layout.addWidget(self.open_frame)
        main_layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignRight)

    def layout_open(self):
        self.open_frame = QtWidgets.QFrame(self)
        self.open_frame.setFrameShape(self.open_frame.StyledPanel)
        self.open_frame.setLineWidth(0.5)
        self.open_frame.setMinimumWidth(600)

        open_label = QtWidgets.QLabel("Select the config file:", self.open_frame)
        self.open_line = QtWidgets.QLineEdit(self.open_frame)
        self.open_line.textChanged[str].connect(self.open_config_name)

        load_button = QtWidgets.QPushButton("Browse")
        load_button.clicked.connect(self.load_config)

        grid = QtWidgets.QGridLayout(self.open_frame)
        grid.setSpacing(30)
        grid.addWidget(open_label, 0, 0)
        grid.addWidget(self.open_line, 0, 1)
        grid.addWidget(load_button, 1, 1)

        return self.open_frame

    def open_config_name(self):
        self.open_line.text()

    def load_config(self):
        cwd = os.getcwd()
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config:
            return
        self.config = config[0]
        self.open_line.setText(self.config)
        self.ok_button.setFocus()

    def open_project(self):
        if self.config == "":
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Please choose the config.yaml file to load the project")

            msg.setWindowTitle("Error")
            msg.setMinimumWidth(400)
            self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
            self.logo = self.logo_dir + "/assets/logo.png"
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

            self.loaded = False
        else:
            self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
            self.logo = self.logo_dir + "/assets/logo.png"

            self.loaded = True
            self.accept()
            self.close()
