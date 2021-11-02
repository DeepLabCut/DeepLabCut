import sys
import deeplabcut
from createStatus import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QTabWidget
from PyQt5.QtWidgets import QWidget, QLabel, QRadioButton, QFormLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QGridLayout, QFrame, QGraphicsDropShadowEffect

from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QSizePolicy, QStatusBar

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap
#import deeplabcut
import os




class MainApp(QWidget):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super().__init__()

        self.setWindowTitle("DeepLabCut")
        self.setMinimumSize(1500, 750)
        #QtWidgets.QMainWindow.setStatusBar(QStatusBar('sb'))

        logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
        logo = logo_dir + '/pitures/logo.png'
        self.setWindowIcon(QIcon(logo))


        # Create a top-level layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Create the tab widget with two tabs
        tabs = QTabWidget()
        tabs.addTab(self.welcomeTabUI(), "Welcome")
        tabs.addTab(self.manage_projectTabUI(), "Manage Project")
        layout.addWidget(tabs)


        self.name_default = ''
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = 'C:/'


    def welcomeTabUI(self):
        """Create the General page UI."""
        welcomeTab = QWidget()
        layout = QVBoxLayout()

        pic_dir = os.path.dirname(os.path.realpath('Welcome.png')) + os.path.sep
        file = pic_dir + '/pitures/Welcome.png'
        pixmap = QPixmap(file) #C:\Users\User\PycharmProjects
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(lbl)
        layout.setAlignment(Qt.AlignTop)

        pol = QSizePolicy()
        pol.setHorizontalPolicy(QSizePolicy.Ignored)
        lbl.setSizePolicy(pol)

        lbl_welcome1 = QLabel("Welcome to the DeepLabCut Project Manager GUI!")
        lbl_welcome1.setAlignment(Qt.AlignCenter)
        lbl_welcome2 = QLabel("To get started, please click on the 'Manage Project'")
        lbl_welcome2.setAlignment(Qt.AlignCenter)
        lbl_welcome3 = QLabel("tab to cteate or load an existing project.")
        lbl_welcome3.setAlignment(Qt.AlignCenter)

        layout.addWidget(lbl_welcome1)
        layout.addWidget(lbl_welcome2)
        layout.addWidget(lbl_welcome3)

        welcomeTab.setLayout(layout)
        return welcomeTab

    def create_or_load(self):

        self.separatorLine = QFrame()
        self.separatorLine.setFrameShape(QFrame.HLine)
        self.separatorLine.setFrameShadow(QFrame.Raised)

        self.separatorLine.setLineWidth(0)
        self.separatorLine.setMidLineWidth(1)

        self.inLayout = QVBoxLayout()
        l1_step1 = QLabel("DeepLabCut - step 1. Create New Project or Load a Project")
        self.inLayout.setSpacing(40)
        self.inLayout.setContentsMargins(0,20,0,40)
        self.inLayout.addWidget(l1_step1)
        self.inLayout.addWidget(self.separatorLine)

    def create_RadioButton(self):
        self.create_layout = QVBoxLayout()
        self.create_layout.setSpacing(20)
        self.create_layout.setContentsMargins(0, 0, 0, 50)

        label = QLabel('Please choose an option:')
        label.setContentsMargins(20, 0, 0, 0)
        label.setAlignment(Qt.AlignTop)
        self.create_layout.addWidget(label)
        self.create_layout.setAlignment(Qt.AlignTop)

        # Create a layout for the RadioButton
        optionsLayout = QHBoxLayout()
        optionsLayout.setSpacing(50)

        create_rb = QRadioButton('Create new project')
        create_rb.setChecked(True)
        #self.create_rb.clicked.connect(self.create_project)

        load_rb = QRadioButton('Load existing project')

        optionsLayout.addWidget(create_rb)
        optionsLayout.addWidget(load_rb)
        optionsLayout.setAlignment(Qt.AlignLeft )

        self.create_layout.addLayout(optionsLayout)



    def names_videos(self):
        self.names_loadvideos_layout = QVBoxLayout()
        self.names_loadvideos_layout.setSpacing(30)
        form_layout = QFormLayout()
        form_layout.maximumSize()
        # Add widgets to the layout
        line1 = QLineEdit()
        line1.setFixedWidth(650)
        #line1.textEdited.connect(self.update_project_name)

        line2 = QLineEdit()
        line2.setFixedWidth(650)
        # line2.textEdited.connect(self.update_experimenter_name)

        form_layout.addRow("Name of the Project:", line1)
        form_layout.addRow("Name of the experimenter:", line2)
        self.names_loadvideos_layout.addLayout(form_layout)

        videosLayout = QHBoxLayout()
        videosLayout.setSpacing(110)

        label = QLabel('Choose Videos:')
        videosLayout.addWidget(label)

        loadbutton = QPushButton('Load Videos')
        loadbutton.setFixedWidth(650)
        loadbutton.clicked.connect(self.load_videos)

        videosLayout.addWidget(loadbutton)
        videosLayout.setAlignment(Qt.AlignLeft)


        self.names_loadvideos_layout.addLayout(videosLayout)

    def load_videos(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a folder',
                                                            self.loc_default)
        if not folder:
            return
        self.loc_default = dirname

    def browse_dir(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a folder', self.loc_default)
        if not dirname:
            return
        self.loc_default = dirname
        #self.update_project_location()
    def optional_attributes(self):
        self.optional_attr = QVBoxLayout()
        self.optional_attr.setSpacing(10)

        # Create a layout for the checkboxes
        #optionsLayout = QVBoxLayout()
        label = QLabel('Optional Attributes')
        label.setContentsMargins(20, 20, 0, 0)
        self.optional_attr.addWidget(label)

        select_dir = QHBoxLayout()
        select_dir.setSpacing(50)
        select_dir.setAlignment(Qt.AlignLeft)

        browse_btn = QPushButton('Browse')
        browse_btn.setFixedWidth(120)
        browse_btn.setEnabled(False)
        #browse_btn.clicked.connect(self.browse_dir)

        select_dir.addWidget(QCheckBox("Select the directory where project will be created"))
        select_dir.addWidget(browse_btn)


        # Add some checkboxes to the layout
        self.optional_attr.addLayout(select_dir)
        self.optional_attr.addWidget(QCheckBox("Copy the videos"))
        self.optional_attr.addWidget(QCheckBox("Is it a multi-animal project?"))

    def manage_project_btns(self):
        self.btns_manage = QGridLayout()
        self.btns_manage.setSpacing(20)

        #l1 = QHBoxLayout()
        #l1.setAlignment(Qt.AlignRight)
        #l1.setSpacing(40)

        # Add widgets to the layout
        btn_load_nv = QPushButton("Load New Videos")
        btn_load_nv.setFixedWidth(320)
        btn_load_nv.setEnabled(False)

        btn_ok = QPushButton("Ok")
        btn_ok.setFixedWidth(120)

        #l1.addWidget(btn_load_nv)
        #l1.addWidget(btn_ok)

        #l2 = QHBoxLayout()
        #l2.setAlignment(Qt.AlignLeft)
        #l2.setSpacing(40)

        # Add widgets to the layout
        btn_help = QPushButton("Help")
        btn_help.setFixedWidth(120)


        btn_reset = QPushButton("Reset")
        btn_reset.setFixedWidth(120)

        btn_add = QPushButton("Add New Videos")
        btn_add.setFixedWidth(220)
        btn_add.setEnabled(False)

        btn_edit = QPushButton("Edit config file")
        btn_edit.setFixedWidth(220)
        btn_edit.setEnabled(False)

        self.btns_manage.addWidget(btn_load_nv, 0, 2)
        self.btns_manage.addWidget(btn_ok, 0, 4)
        self.btns_manage.addWidget(btn_help, 1, 0)
        self.btns_manage.addWidget(btn_reset, 1, 1)
        self.btns_manage.addWidget(btn_add, 1, 2)
        self.btns_manage.addWidget(btn_edit, 1, 3)

        #l2.addWidget(btn_help)
        #l2.addWidget(btn_reset)
        #l2.addWidget(btn_add)
        #l2.addWidget(btn_edit)

        #self.btns_manage.addLayout(l1)
        #self.btns_manage.addLayout(l2)

    def manage_projectTabUI(self):
        """Create the Manage project page UI."""
        manage_projectTab = QWidget()
        main_layout = QVBoxLayout()

        self.create_or_load()
        self.create_RadioButton()
        self.names_videos()
        self.optional_attributes()
        self.manage_project_btns()

        main_layout.addLayout(self.inLayout)
        main_layout.addLayout(self.create_layout)
        main_layout.addLayout(self.names_loadvideos_layout)
        main_layout.addLayout(self.optional_attr)
        main_layout.addLayout(self.btns_manage)

        main_layout.setAlignment(Qt.AlignTop)
        manage_projectTab.setLayout(main_layout)
        return manage_projectTab

    def set_window_layout(self):
        main_vertical_layout = QVBoxLayout(self.centralwidget)
        main_vertical_layout.addLayout(self.inLayout)
        main_vertical_layout.addLayout(self.create_layout)
        main_vertical_layout.setAlignment(Qt.AlignTop)

        return main_vertical_layout



