
import deeplabcut

from PyQt5 import QtCore, QtWidgets
import os

class CreateProject(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle('New Project')
        self.setMinimumSize(800, 400)
        #self.create_window_set()

        #today, _ =
        #deeplabcut.create_project.format_date()
        self.name_default = '-'.join(('{}', '{}', 'newProject'))
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = parent.project_folder

        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_user()
        #self.video_frame = self.lay_out_video_frame()
        self.create_button = QtWidgets.QPushButton('Create')
        self.create_button.setDefault(True)
        # self.create_button.clicked.connect(self.finalize_project)
        main_layout.addWidget(self.user_frame)
        #main_layout.addWidget(self.video_frame)
        main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def layout_user(self):
        self.user_frame = QtWidgets.QFrame(self)
        self.user_frame.setFrameShape(self.user_frame.StyledPanel)
        self.user_frame.setLineWidth(0.5)

        proj_label = QtWidgets.QLabel('Name of the project:', self.user_frame)
        proj_line = QtWidgets.QLineEdit(self.proj_default, self.user_frame)
        self._default_style = proj_line.styleSheet()
        # self.proj_line.textEdited.connect(self.update_project_name)

        exp_label = QtWidgets.QLabel('Name of the experimenter:', self.user_frame)
        self.exp_line = QtWidgets.QLineEdit(self.exp_default, self.user_frame)
        # self.exp_line.textEdited.connect(self.update_experimenter_name)

        videos_label = QtWidgets.QLabel('Choose Videos:', self.user_frame)
        load_button = QtWidgets.QPushButton('Load Videos')
        load_button.clicked.connect((self.on_click))

        grid = QtWidgets.QGridLayout(self.user_frame)
        grid.setSpacing(30)
        grid.addWidget(proj_label, 0, 0)
        grid.addWidget(proj_line, 0, 1)
        grid.addWidget(exp_label, 1, 0)
        grid.addWidget(self.exp_line, 1, 1)
        grid.addWidget(videos_label, 2, 0)
        grid.addWidget(load_button, 2, 1)

        return self.user_frame

    def on_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a folder', self.loc_default)
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        self.loc_default = dirname
        #self.update_project_location()

    def update_project_location(self):
        full_name = self.name_default.format(self.proj_default, self.exp_default)
        full_path = os.path.join(self.loc_default, full_name)
        self.loc_line.setText(full_path)
