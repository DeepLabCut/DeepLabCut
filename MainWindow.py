import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QTabWidget
from PyQt5.QtWidgets import QWidget, QLabel, QRadioButton, QFormLayout, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGridLayout, QFrame, QGraphicsDropShadowEffect, QMenu

from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QSizePolicy, QStatusBar

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui


#from MainApp import *
from CreateProject import *
from OpenProject import *
from extract_frames import *
from label_frames import *
from create_training_dataset import *
from train_network import *
from evaluate_network import *
from video_editor import *
from analyze_videos import  *
from create_videos import *
from extract_outlier_frames import *
from refine_labels import *

import deeplabcut
#from deeplabcut.gui import canvas, widgets
#from deeplabcut.utils import auxiliaryfunctions, video_reader
import os
# import Dark Mode Theme
import qdarkstyle
#import breeze_resources

#import qrc_resources


class MainWindow(QtWidgets.QMainWindow):
    #config_loaded = QtCore.pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()

        desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        self.screen_width = desktop.width()
        self.screen_height = desktop.height()
        self.config = None
        self.cfg = dict()
        self.loaded = False
        self.user_feedback = False

        self.default_set()

        self.welcome_page('Welcome.png')
        self.window_set()
        self.default_set()


        names = ['new_project.png', 'open.png', 'help.png']
        self.createActions(names)
        self.createMenuBar()
        self.createToolBars(0)




        #self.canvas = canvas.Canvas(self.main_panel)
        #self.figtitle = widgets.ClickableLabel()
        #self.figtitle.setAlignment(QtCore.Qt.AlignCenter)
        #self.figtitle.setEnabled(False)


    def window_set(self):
        self.setWindowTitle("DeepLabCut")
        #self.setGeometry(300,150,1500,750)
        self.setMinimumSize(1500, 750)
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("www.deeplabcut.org")

        self.setStatusBar(self.statusbar)


        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        self.logo_dir = os.path.dirname(os.path.realpath('logo.png')) + os.path.sep
        self.logo = self.logo_dir + '/pictures/logo.png'
        self.setWindowIcon(QIcon(self.logo))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName('Status Bar')

    def set_pic(self, name):
        pic_dir = os.path.dirname(os.path.realpath(name)) + os.path.sep
        file = pic_dir + '/pictures/' + name
        pixmap = QPixmap(file)  # C:\Users\User\PycharmProjects
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lbl.setScaledContents(True)
        lbl.setMaximumWidth(1565)
        return lbl

    def welcome_page(self, name):
        layout = QVBoxLayout()

        layout.addWidget(self.set_pic(name))
        layout.setAlignment(Qt.AlignTop)

        #pol = QSizePolicy()
        #pol.setHorizontalPolicy(QSizePolicy.Ignored)
        #lbl.setSizePolicy(pol)

        lbl_welcome1 = QLabel("Welcome to the DeepLabCut Project Manager GUI!")
        lbl_welcome1.setAlignment(Qt.AlignCenter)
        lbl_welcome2 = QLabel("To get started, please click on the 'File'")
        lbl_welcome2.setAlignment(Qt.AlignCenter)
        lbl_welcome3 = QLabel("tab to cteate or load an existing project.")
        lbl_welcome3.setAlignment(Qt.AlignCenter)

        layout.addWidget(lbl_welcome1)
        layout.addWidget(lbl_welcome2)
        layout.addWidget(lbl_welcome3)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        #desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        #self.screen_width = desktop.width()
        #self.screen_height = desktop.height()
        #self.setFocus()
        #self.activateWindow()

    def project_folder(self):
        return self.cfg.get('project_path', os.path.expanduser('~/Desktop'))

    def default_set(self):
        self.name_default = ''
        self.proj_default = ''
        self.exp_default = ''
        self.loc_default = 'C:/'


    def createActions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")
        #self.newAction.setToolTip('Create new project')

        self.newAction.setIcon(QIcon("icons/"+names[0]))
        self.newAction.setShortcut('Ctrl+N')

        self.newAction.triggered.connect(self._create)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(QIcon("icons/"+names[1]))
        self.openAction.setShortcut('Ctrl+O')
        self.openAction.triggered.connect(self._open)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(QIcon("icons/"+names[2]))

        self.aboutAction = QAction("&Learn DLC", self)


    def createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        self.fileMenu = QMenu("&File", self)
        menuBar.addMenu(self.fileMenu)


        self.fileMenu.addAction(self.newAction)
        self.fileMenu.addAction(self.openAction)

        findMenu = self.fileMenu.addMenu("Open Recent")
        findMenu.addAction("File 1")
        findMenu.addAction("File 2")
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.exitAction)
        # View menu
        viewMenu = QMenu("&View", self)
        mode = viewMenu.addMenu("Appearance")
        menuBar.addMenu(viewMenu)
        mode.addAction(self.lightmodeAction)
        mode.addAction(self.darkmodeAction)

        # Help menu
        helpMenu = QMenu("&Help", self)
        menuBar.addMenu(helpMenu)
        helpMenu.addAction(self.helpAction)
        helpMenu.adjustSize()
        helpMenu.addAction(self.aboutAction)

    def updateMenuBar(self):
        self.fileMenu.removeAction(self.newAction)
        self.fileMenu.removeAction(self.openAction)
        #self.fileMenu.addAction(self.helpAction)
        #helpMenu.adjustSize()

        #self.fileMenu.addAction(self.newAction)
        #self.fileMenu.addAction(self.openAction)

    def createToolBars(self, flag):
        # File toolbar
        if flag == 0:
            self.fileToolBar = self.addToolBar("File")

        self.fileToolBar.addAction(self.newAction)
        self.fileToolBar.addAction(self.openAction)
        self.fileToolBar.addAction(self.helpAction)
    def remove_action(self):
        self.fileToolBar.removeAction(self.newAction)
        self.fileToolBar.removeAction(self.openAction)
        self.fileToolBar.removeAction(self.helpAction)


    @QtCore.pyqtSlot()
    def _create(self):
        # action = self.sender()
        create_p = CreateProject(self)
        create_p.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if create_p.exec_() == QtWidgets.QDialog.Accepted:
            #print('loaded:   ', create_p.loaded)
            self.loaded =create_p.loaded
            #print('cfg:   ', create_p.cfg)
            self.cfg = create_p.cfg
            self.user_feedback = create_p.user_fbk

        if create_p.loaded:
            self.add_tabs()

    def _open(self):
        open_p = OpenProject(self)
        open_p.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if open_p.exec_() == QtWidgets.QDialog.Accepted:
            print('loaded:   ', open_p.loaded)
            self.loaded = open_p.loaded
            print('cfg:   ', open_p.cfg)
            self.cfg = open_p.cfg
            self.user_feedback = open_p.user_fbk

        if open_p.loaded:
            self.add_tabs()


    def load_config(self, config):
        self.config = config
        self.cfg = auxiliaryfunctions.read_config(config)
        self.config_loaded.emit()
        print(f'Project "{self.cfg["Task"]}" successfully loaded.')

    def darkmode(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.welcome_page('Welcome2.png')

        names = ['new_project2.png', 'open2.png', 'help2.png']
        self.remove_action()
        self.createActions(names)
        self.updateMenuBar()
        self.createToolBars(1)
    def lightmode(self):
        self.setStyleSheet('Fusion')
        self.welcome_page('Welcome.png')

        names = ['new_project.png', 'open.png', 'help.png']
        self.remove_action()
        self.createActions(names)
        self.createToolBars(1)
        self.updateMenuBar()

    def add_tabs(self):
        # Add all the other pages
        # Create a top-level layout

        # Create the tab widget with two tabs
        tabs = QtWidgets.QTabWidget()
        tabs.setContentsMargins(0, 20, 0, 0)
        extract_page = Extract_page(self, self.cfg)
        label_page = Label_page(self, self.cfg)
        create_training_ds_page = Create_training_dataset_page(self, self.cfg)
        train_network_page = Train_network_page(self, self.cfg)
        evaluate_network_page = Evaluate_network_page(self, self.cfg)
        video_editor_page = Video_editor_page(self, self.cfg)
        analyze_videos_page = Analyze_videos_page(self, self.cfg)
        create_videos_page = Create_videos_page(self, self.cfg)
        extract_outlier_frames_page = Extract_outlier_frames_page(self, self.cfg)
        refine_labels_page = Refine_labels_page(self, self.cfg)

        tabs.addTab(extract_page, "Extract frames")
        tabs.addTab(label_page, "Label frames")
        tabs.addTab(create_training_ds_page, "Create training dataset")
        tabs.addTab(train_network_page, "Train network")
        tabs.addTab(evaluate_network_page, "Evaluate network")
        tabs.addTab(video_editor_page, "Video editor")
        tabs.addTab(analyze_videos_page, "Analyze videos")
        tabs.addTab(create_videos_page, "Create videos")
        tabs.addTab(extract_outlier_frames_page, "Extract outlier frames")
        tabs.addTab(refine_labels_page, "Refine labels")

        self.setCentralWidget(tabs)

