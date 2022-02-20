
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox
import os

class OpenProject(QtWidgets.QDialog):
    def __init__(self, parent):
        super(OpenProject, self).__init__(parent)

        self.setWindowTitle('Load Existing Project')
        self.setMinimumSize(800, 400)

        self.cfg = None
        self.loaded = False
        self.user_fbk = True

        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_open()

        self.open_button = QtWidgets.QPushButton('Ok')
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.open_project)

        main_layout.addWidget(self.open_frame)
        main_layout.addWidget(self.open_button, alignment=QtCore.Qt.AlignRight)


    def layout_open(self):
        self.open_frame = QtWidgets.QFrame(self)
        self.open_frame.setFrameShape(self.open_frame.StyledPanel)
        self.open_frame.setLineWidth(0.5)

        open_label = QtWidgets.QLabel('Select the config file:', self.open_frame)
        self.open_line = QtWidgets.QLineEdit(self.open_frame)
        self.open_line.textChanged[str].connect(self.open_config_name)

        load_button = QtWidgets.QPushButton('Browse')
        load_button.clicked.connect((self.load_config))

        label = QtWidgets.QLabel('Optional Attributes:')

        ch_box = QCheckBox("User feedback")
        ch_box.stateChanged.connect(self.activate_fbk)

        grid = QtWidgets.QGridLayout(self.open_frame)
        grid.setSpacing(30)
        grid.addWidget(open_label, 0, 0)
        grid.addWidget(self.open_line, 0, 1)
        grid.addWidget(load_button, 1, 1)
        grid.addWidget(label, 2, 0)
        grid.addWidget(ch_box, 2, 1)

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
        self.cfg = config[0]
        self.open_line.setText(self.cfg)

    def activate_fbk(self, state):
        # Activates the feedback option
        # TODO: finish functionality: with user feedback (self.user_fbk = True) / without (self.user_fbk = False)
        if state == QtCore.Qt.Checked:
            self.user_fbk = True
        else:
            self.user_fbk = False

    def open_project(self):
        if self.cfg == '':
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Please choose the config.yaml file to load the project")

            msg.setWindowTitle("Error")
            msg.setMinimumWidth(400)
            self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
            self.logo = self.logo_dir + '/pictures/logo.png'
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()

            self.loaded = False
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Project Loaded!")

            msg.setWindowTitle("Info")
            msg.setMinimumWidth(400)
            self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
            self.logo = self.logo_dir + '/pictures/logo.png'
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.buttonClicked.connect(self.ok_clicked)
            msg.exec_()
            self.loaded = True

            self.close()

    def ok_clicked(self):
            self.loaded = True
            self.accept()



