
import deeplabcut
from deeplabcut.create_project import create_new_project, add_new_videos
from deeplabcut.gui.analyze_videos import Analyze_videos
from deeplabcut.gui.create_training_dataset import Create_training_dataset
from deeplabcut.gui.create_videos import Create_Labeled_Videos
from deeplabcut.gui.evaluate_network import Evaluate_network
from deeplabcut.gui.extract_frames import Extract_frames
from deeplabcut.gui.extract_outlier_frames import Extract_outlier_frames
from deeplabcut.gui.label_frames import Label_frames
from deeplabcut.gui.refine_labels import Refine_labels
from deeplabcut.gui.refine_tracklets import Refine_tracklets
from deeplabcut.gui.train_network import Train_network
from deeplabcut.gui.video_editing import Video_Editing
from deeplabcut.utils import auxiliaryfunctions


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox, QButtonGroup
from PyQt5.QtCore import Qt
import os

class CreateProject(QtWidgets.QDialog):
    def __init__(self, parent):
        super(CreateProject, self).__init__(parent)

        self.setWindowTitle('New Project')
        self.setMinimumSize(900, 500)

        #self.ok_clicked()

        #self.create_window_set()

        #today, _ =
        #deeplabcut.create_project.format_date()
        self.name_default = '-'.join(('{}', '{}', 'newProject'))
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = ''

        self.cfg = None
        self.copy = False
        self.loaded = False
        self.user_fbk = True
        self.filelist = []


        main_layout = QtWidgets.QVBoxLayout(self)
        self.layout_user()
        #self.video_frame = self.lay_out_video_frame()

        self.create_button = QtWidgets.QPushButton('Create')
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self.create_newproject)
        main_layout.addWidget(self.user_frame)
        #main_layout.addWidget(self.video_frame)
        main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def layout_user(self):
        self.user_frame = QtWidgets.QFrame(self)
        self.user_frame.setFrameShape(self.user_frame.StyledPanel)
        self.user_frame.setLineWidth(0.5)

        proj_label = QtWidgets.QLabel('Name of the project:', self.user_frame)
        self.proj_line = QtWidgets.QLineEdit(self.user_frame)
        self.proj_line.textChanged[str].connect(self.update_project_name)
        #self._default_style = proj_line.styleSheet()
        #self.proj_line.textEdited.connect(self.update_project_name)

        exp_label = QtWidgets.QLabel('Name of the experimenter:', self.user_frame)
        self.exp_line = QtWidgets.QLineEdit(self.user_frame)
        self.exp_line.textChanged[str].connect(self.update_experimenter_name)
        # self.exp_line.textEdited.connect(self.update_experimenter_name)

        videos_label = QtWidgets.QLabel('Choose Videos:', self.user_frame)
        self.load_button = QtWidgets.QPushButton('Load Videos')
        self.load_button.clicked.connect((self.load_videos))




        grid = QtWidgets.QGridLayout(self.user_frame)
        grid.setSpacing(30)
        grid.addWidget(proj_label, 0, 0)
        grid.addWidget(self.proj_line, 0, 1)
        grid.addWidget(exp_label, 1, 0)
        grid.addWidget(self.exp_line, 1, 1)
        grid.addWidget(videos_label, 2, 0)
        grid.addWidget(self.load_button, 2, 1)


        #optional_attr = QtWidgets.QGridLayout()
        #optional_attr.setAlignment(Qt.AlignTop)

        # Create a layout for the checkboxes

        label = QtWidgets.QLabel('Optional Attributes:')
        #label.setContentsMargins(0, 20, 0, 0)
        grid.addWidget(label, 3,0)
        # Add some checkboxes to the layout
        ch_box1 = QCheckBox("Select the directory where project will be created")
        ch_box1.stateChanged.connect(self.activate_browse)

        self.browse_button = QtWidgets.QPushButton('Browse')
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect((self.browse_dir))

        grid.addWidget(ch_box1,4,0)
        grid.addWidget(self.browse_button,4,1)


        ch_box2 = QCheckBox("Copy the videos")
        ch_box2.stateChanged.connect(self.activate_copy_videos)

        ch_box3 = QCheckBox("Is it a multi-animal project?")
        self.multi_choice = False
        #ch_box3.stateChanged.connect(self.clickBox)

        ch_box4 = QCheckBox("User feedback")
        ch_box4.stateChanged.connect(self.activate_fbk)

        #optional_attr.addLayout(sel_dir, alignment=QtCore.Qt.AlignTop)
        grid.addWidget(ch_box2, 5, 0)
        grid.addWidget(ch_box3, 6, 0)
        grid.addWidget(ch_box4, 7, 0)

        # label = QtWidgets.QLabel('Do you want feedback?')
        # # label.setContentsMargins(0, 20, 0, 0)
        #
        # fbk_layout = QtWidgets.QHBoxLayout()
        # fbk_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # fbk_layout.setSpacing(0)
        # fbk_layout.setContentsMargins(20, 0, 0, 0)
        #
        # self.btngroup_opencv = QButtonGroup()
        #
        # self.fbk_choice1 = QtWidgets.QRadioButton('Yes')
        # self.fbk_choice1.setChecked(True)
        # #self.fbk_choice1.toggled.connect(lambda: self.update_feedback_choice(self.fbk_choice1))
        #
        # self.fbk_choice2 = QtWidgets.QRadioButton('No')
        # #self.fbk_choice2.toggled.connect(lambda: self.update_feedback_choice(self.fbk_choice2))
        #
        # self.btngroup_opencv.addButton(self.fbk_choice1)
        # self.btngroup_opencv.addButton(self.fbk_choice2)
        #
        # fbk_layout.addWidget(self.fbk_choice1)
        # fbk_layout.addWidget(self.fbk_choice2)
        #
        # grid.addWidget(label, 7, 0)
        # grid.addLayout(fbk_layout, 7, 1)



        return self.user_frame

    def on_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a folder', self.loc_default)
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        self.loc_default = dirname
        #self.update_project_location()

    def update_project_name(self):
        text = self.proj_line.text()
        self.proj_default = text
        #self.update_project_location()

    def update_experimenter_name(self):
        text = self.exp_line.text()
        self.exp_default = text
        #self.update_project_location()

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
        #cwd = os.getcwd()
        cwd = 'C:/Anna/'
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self,
                    "Choose the directory where your project will be saved:",
                    cwd)
        if not dirname:
            return
        dirname = QtCore.QDir.toNativeSeparators(dirname)
        #self.dir = dirname
        self.loc_default = dirname
        print(self.loc_default)
        #self.update_project_location()



    def load_videos(self):
        #cwd = os.getcwd()
        cwd = 'C:/Users/User/DeepLabCut/examples/openfield-Pranav-2018-10-30/videos/'
        videos_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select videos to add to the project", cwd, "", "*.*"
        )
        self.videos = videos_file[0]
        self.filelist.append(self.videos)
        #num_of_videos = "Total %s Videos selected" % len(self.filelist)
        self.load_button.setText("Total %s Videos selected"% len(self.filelist))

    def activate_fbk(self, state):
        # Activates the feedback option
        if state == QtCore.Qt.Checked:
            self.user_fbk = True
            print('T')
        else:
            self.user_fbk = False
            print('F')

    def update_project_location(self):
        full_name = self.name_default.format(self.proj_line.text(), self.exp_line.text())
        full_path = os.path.join(self.loc_default, full_name)
        self.loc_pr = full_path
        print(full_path)


    def create_newproject(self):
        # create the new project
        if self.proj_default != '' and self.exp_default != '' and self.filelist != []:
            # print('hi')
            # print(self.proj_default)
            # print(self.exp_default)
            # print(self.filelist)
            # print(self.loc_default)
            # print(self.copy)
            # self.cfg = create_new_project('reaching-task','Linus',
            #              ['C:/Users/User/DeepLabCut/examples/openfield-Pranav-2018-10-30/videos/m3v1mp4.mp4'],
            #               working_directory = 'C:/Anna',
            #               copy_videos=True)
            self.cfg = create_new_project(
                self.proj_default,
                self.exp_default,
                self.filelist,
                self.loc_default,
                copy_videos=self.copy
            )
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Some of the enteries are missing.")
            msg.setInformativeText("Make sure that the task and experimenter name are specified and videos are selected!")
            msg.setWindowTitle("Error")
            msg.setMinimumWidth(300)
            #msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)

            retval = msg.exec_()
            self.cfg = False

        if self.cfg:
            #print('cfg')
            self.loaded = True

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("New Project Created")

            msg.setWindowTitle("Info")
            msg.setMinimumWidth(400)
            self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
            self.logo = self.logo_dir + '/pictures/logo.png'
            msg.setWindowIcon(QIcon(self.logo))
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.buttonClicked.connect(self.ok_clicked)
            retval = msg.exec_()

            self.close()
            #self.edit_config_file.Enable(True)
        # Add all the other pages
        # if self.loaded:
        #     cfg = auxiliaryfunctions.read_config(self.cfg)
        #     print(self.loaded)
    def ok_clicked(self):
        #print('ok')
        self.loaded = True
        self.accept()





