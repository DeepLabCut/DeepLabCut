import os
from datetime import datetime

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.widgets import ClickableLabel, ItemSelectionFrame

from PySide2 import QtCore, QtWidgets


class ProjectCreator(QtWidgets.QDialog):
    def __init__(self, parent):
        super(ProjectCreator, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle('New Project')
        self.setMinimumWidth(parent.screen_width // 2)
        today = datetime.today().strftime('%Y-%m-%d')
        self.name_default = '-'.join(('{}', '{}', today))
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = parent.project_folder

        main_layout = QtWidgets.QVBoxLayout(self)
        self.user_frame = self.lay_out_user_frame()
        self.video_frame = self.lay_out_video_frame()
        self.create_button = QtWidgets.QPushButton('Create')
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self.finalize_project)
        main_layout.addWidget(self.user_frame)
        main_layout.addWidget(self.video_frame)
        main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def lay_out_user_frame(self):
        user_frame = QtWidgets.QFrame(self)
        user_frame.setFrameShape(user_frame.StyledPanel)
        user_frame.setLineWidth(0.5)

        proj_label = QtWidgets.QLabel('Project:', user_frame)
        self.proj_line = QtWidgets.QLineEdit(self.proj_default, user_frame)
        self._default_style = self.proj_line.styleSheet()
        self.proj_line.textEdited.connect(self.update_project_name)

        exp_label = QtWidgets.QLabel('Experimenter:', user_frame)
        self.exp_line = QtWidgets.QLineEdit(self.exp_default, user_frame)
        self.exp_line.textEdited.connect(self.update_experimenter_name)

        loc_label = ClickableLabel('Location:', parent=user_frame)
        loc_label.signal.connect(self.on_click)
        self.loc_line = QtWidgets.QLineEdit(self.loc_default, user_frame)

        vbox = QtWidgets.QVBoxLayout(user_frame)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(proj_label, 0, 0)
        grid.addWidget(self.proj_line, 0, 1)
        grid.addWidget(exp_label, 1, 0)
        grid.addWidget(self.exp_line, 1, 1)
        grid.addWidget(loc_label, 2, 0)
        grid.addWidget(self.loc_line, 2, 1)
        vbox.addLayout(grid)

        self.madlc_box = QtWidgets.QCheckBox('Is it a multi-animal project?')
        self.madlc_box.setChecked(False)
        vbox.addWidget(self.madlc_box)

        return user_frame

    def lay_out_video_frame(self):
        video_frame = ItemSelectionFrame([], self)

        self.cam_combo = QtWidgets.QComboBox(video_frame)
        self.cam_combo.addItems(map(str, (1, 2)))
        self.cam_combo.currentTextChanged.connect(self.check_num_cameras)
        ncam_label = QtWidgets.QLabel('Number of cameras:')
        ncam_label.setBuddy(self.cam_combo)

        self.copy_box = QtWidgets.QCheckBox('Copy videos to project folder')
        self.copy_box.setChecked(False)

        browse_button = QtWidgets.QPushButton('Browse videos')
        browse_button.clicked.connect(self.browse_videos)
        clear_button = QtWidgets.QPushButton('Clear')
        clear_button.clicked.connect(video_frame.fancy_list.clear)

        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(ncam_label)
        layout1.addWidget(self.cam_combo)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(browse_button)
        layout2.addWidget(clear_button)
        video_frame.layout.insertLayout(0, layout1)
        video_frame.layout.addLayout(layout2)
        video_frame.layout.addWidget(self.copy_box)

        return video_frame

    def browse_videos(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Please select a folder', self.loc_default,
        )
        if not folder:
            return
        for video in auxiliaryfunctions.grab_files_in_folder(
                folder, relative=False,
        ):
            if video.split('.')[1] in DLCParams.VIDEOTYPES[1:]:
                self.video_frame.fancy_list.add_item(video)

    def finalize_project(self):
        fields = [self.proj_line, self.exp_line]
        empty = [i for i, field in enumerate(fields) if not field.text()]
        for i, field in enumerate(fields):
            if i in empty:
                field.setStyleSheet('border: 1px solid red;')
            else:
                field.setStyleSheet(self._default_style)
        if empty:
            return

        n_cameras = int(self.cam_combo.currentText())
        try:
            if n_cameras > 1:
                _ = deeplabcut.create_new_project_3d(
                    self.proj_default,
                    self.exp_default,
                    n_cameras,
                    self.loc_default,
                )
            else:
                videos = list(self.video_frame.selected_items)
                if not len(videos):
                    print('Add at least a video to the project.')
                    self.video_frame.fancy_list.setStyleSheet('border: 1px solid red')
                    return
                else:
                    self.video_frame.fancy_list.setStyleSheet(
                        self.video_frame.fancy_list._default_style
                    )
                to_copy = self.copy_box.isChecked()
                is_madlc = self.madlc_box.isChecked()
                config = deeplabcut.create_new_project(
                    self.proj_default,
                    self.exp_default,
                    videos,
                    self.loc_default,
                    to_copy,
                    multianimal=is_madlc,
                )
                self.parent.load_config(config)
                self.parent._update_project_state(
                    config=config,
                    loaded=True,
                )
        except FileExistsError:
            print('Project "{}" already exists!'.format(self.proj_default))
            return

        msg = QtWidgets.QMessageBox(text=f"New project created")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()

        self.close()

    def on_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a folder', self.loc_default)
        if not dirname:
            return
        self.loc_default = dirname
        self.update_project_location()

    def check_num_cameras(self, value):
        val = int(value)
        for child in self.video_frame.children():
            if child.isWidgetType() and not isinstance(child, QtWidgets.QComboBox):
                if val > 1:
                    child.setDisabled(True)
                else:
                    child.setDisabled(False)

    def update_project_name(self, text):
        self.proj_default = text
        self.update_project_location()

    def update_experimenter_name(self, text):
        self.exp_default = text
        self.update_project_location()

    def update_project_location(self):
        full_name = self.name_default.format(self.proj_default, self.exp_default)
        full_path = os.path.join(self.loc_default, full_name)
        self.loc_line.setText(full_path)
