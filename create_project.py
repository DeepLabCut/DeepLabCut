from deeplabcut.create_project import create_new_project

from PySide2 import QtCore, QtWidgets
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QCheckBox
import os


class CreateProject(QtWidgets.QDialog):
    def __init__(self, parent):
        super(CreateProject, self).__init__(parent)

        self.setWindowTitle("New Project")

        self.name_default = "-".join(("{}", "{}", "newProject"))
        self.proj_default = ""
        self.exp_default = ""
        self.loc_default = ""
        self.project_location = ""

        self.config = None
        self.copy = False
        self.loaded = False
        self.user_fbk = True
        self.filelist = []

        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_user()

        self.create_button = QtWidgets.QPushButton("Create")
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self.create_new_project)
        main_layout.addWidget(self.user_frame)
        main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def layout_user(self):
        self.user_frame = QtWidgets.QFrame(self)
        self.user_frame.setFrameShape(self.user_frame.StyledPanel)
        self.user_frame.setLineWidth(0.5)

        proj_label = QtWidgets.QLabel("Name of the project:", self.user_frame)
        self.proj_line = QtWidgets.QLineEdit(self.user_frame)
        self.proj_line.textChanged[str].connect(self.update_project_name)

        exp_label = QtWidgets.QLabel("Name of the experimenter:", self.user_frame)
        self.exp_line = QtWidgets.QLineEdit(self.user_frame)
        self.exp_line.textChanged[str].connect(self.update_experimenter_name)

        videos_label = QtWidgets.QLabel("Choose Videos:", self.user_frame)
        self.load_button = QtWidgets.QPushButton("Load Videos")
        self.load_button.clicked.connect((self.load_videos))

        grid = QtWidgets.QGridLayout(self.user_frame)
        grid.setSpacing(30)
        grid.addWidget(proj_label, 0, 0)
        grid.addWidget(self.proj_line, 0, 1)
        grid.addWidget(exp_label, 1, 0)
        grid.addWidget(self.exp_line, 1, 1)
        grid.addWidget(videos_label, 2, 0)
        grid.addWidget(self.load_button, 2, 1)

        # Create a layout for the checkboxes
        label = QtWidgets.QLabel("Optional Attributes:")
        grid.addWidget(label, 3, 0)

        # Add some checkboxes to the layout
        ch_box1 = QCheckBox("Select the directory where project will be created")
        ch_box1.stateChanged.connect(self.activate_browse)

        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect((self.browse_dir))

        grid.addWidget(ch_box1, 4, 0)
        grid.addWidget(self.browse_button, 4, 1)

        ch_box2 = QCheckBox("Copy the videos")
        ch_box2.stateChanged.connect(self.activate_copy_videos)

        ch_box3 = QCheckBox("Is it a multi-animal project?")
        self.multi_choice = False

        ch_box4 = QCheckBox("User feedback")
        ch_box4.stateChanged.connect(self.activate_fbk)

        grid.addWidget(ch_box2, 5, 0)
        grid.addWidget(ch_box3, 6, 0)
        grid.addWidget(ch_box4, 7, 0)

        return self.user_frame

    def on_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Please select a folder", self.loc_default
        )
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        self.loc_default = dirname

    def update_project_name(self):
        text = self.proj_line.text()
        self.proj_default = text

    def update_experimenter_name(self):
        text = self.exp_line.text()
        self.exp_default = text

    def activate_browse(self, state):
        # Activates the option to change the working directory
        if state == QtCore.Qt.Checked:
            self.browse_button.setEnabled(True)
        else:
            self.browse_button.setEnabled(False)

    def activate_copy_videos(self, state):
        # Activates the option to copy videos
        if state == QtCore.Qt.Checked:
            self.copy = True
        else:
            self.copy = False

    def browse_dir(self):
        cwd = os.getcwd()
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the directory where your project will be saved:", cwd
        )
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        self.loc_default = dirname

    def load_videos(self):
        cwd = os.getcwd()
        videos_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select videos to add to the project", cwd, "", "*.*"
        )
        self.videos = videos_file[0]
        self.filelist.append(self.videos)
        self.load_button.setText("Total %s Videos selected" % len(self.filelist))

    def activate_fbk(self, state):
        # Activates the feedback option
        if state == QtCore.Qt.Checked:
            self.user_fbk = True
        else:
            self.user_fbk = False
        # TODO: finish functionality: with user feedback (self.user_fbk = True) / without (self.user_fbk = False)

    def update_project_location(self):
        full_name = self.name_default.format(
            self.proj_line.text(), self.exp_line.text()
        )
        full_path = os.path.join(self.loc_default, full_name)
        self.project_location = full_path

    def create_new_project(self):
        # create the new project
        if self.proj_default != "" and self.exp_default != "" and self.filelist != []:
            self.config = create_new_project(
                self.proj_default,
                self.exp_default,
                self.filelist,
                self.loc_default,
                copy_videos=self.copy,
            )

            # self.update_project_location()

        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Some of the entries are missing.")
            msg.setInformativeText(
                "Make sure that the task and experimenter name are specified and videos are selected!"
            )
            msg.setWindowTitle("Error")
            msg.setMinimumWidth(300)
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            self.config = False

        if self.config:
            self.loaded = True

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("New Project Created")

            msg.setWindowTitle("Info")
            msg.setMinimumWidth(400)
            self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
            self.logo = self.logo_dir + "/assets/logo.png"
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.buttonClicked.connect(self.ok_clicked)
            msg.exec_()

            self.close()

    def ok_clicked(self):
        self.loaded = True
        self.accept()
