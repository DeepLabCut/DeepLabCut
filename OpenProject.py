import deeplabcut

from PyQt5 import QtWidgets
import os

class OpenProject(QtWidgets.QDialog):
    def __init__(self, loaded):
        super().__init__()

        self.setWindowTitle('Load Existing Project')
        self.setMinimumSize(800, 400)

        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_open()

        #self.create_button = QtWidgets.QPushButton('Create')
        #self.create_button.setDefault(True)
        # self.create_button.clicked.connect(self.finalize_project)
        main_layout.addWidget(self.open_frame)
        self.exec_()
        #main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def layout_open(self):
        self.open_frame = QtWidgets.QFrame(self)
        self.open_frame.setFrameShape(self.open_frame.StyledPanel)
        self.open_frame.setLineWidth(0.5)

        open_label = QtWidgets.QLabel('Select the config file:', self.open_frame)
        self.open_line = QtWidgets.QLineEdit(self.open_frame)
        self.open_line.textChanged[str].connect(self.open_config_name)

        load_button = QtWidgets.QPushButton('Browse')
        load_button.clicked.connect((self.load_config))

        grid = QtWidgets.QGridLayout(self.open_frame)
        grid.setSpacing(30)
        grid.addWidget(open_label, 0, 0)
        grid.addWidget(self.open_line, 0, 1)

        grid.addWidget(load_button, 1, 1)

        return self.open_frame

    def open_config_name(self):
        #self.proj_default = text
        text = self.open_line.text()
        print(text)

    def load_config(self):
        cwd = os.getcwd()
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config:
            return
        #self._config_file(config)

    #def _config_file(self,):


