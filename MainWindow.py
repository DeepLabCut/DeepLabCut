import os
import logging
from pathlib import Path
from typing import List
import qdarkstyle

from deeplabcut import auxiliaryfunctions

from PySide2.QtWidgets import QAction, QMenu, QLabel, QVBoxLayout, QWidget, QMainWindow
from PySide2 import QtCore
from PySide2.QtGui import QPixmap, QIcon
from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import Qt

from create_project import CreateProject
from open_project import OpenProject
from extract_frames import ExtractFrames
from label_frames import LabelFrames
from create_training_dataset import CreateTrainingDataset
from train_network import TrainNetwork
from evaluate_network import EvaluateNetwork
from video_editor import VideoEditor
from analyze_videos import AnalyzeVideos
from create_videos import CreateVideos
from extract_outlier_frames import ExtractOutlierFrames
from refine_labels import RefineLabels


class MainWindow(QMainWindow):

    config_loaded = QtCore.Signal() 

    def __init__(self):
        super(MainWindow, self).__init__()

        desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        self.screen_width = desktop.width()
        self.screen_height = desktop.height()

        self.logger = logging.getLogger("GUI")

        self.config = None
        self.loaded = False
        self.user_feedback = False

        self.shuffle_value = 1

        self.selected_files = set()

        self.default_set()

        self.welcome_page("Welcome.png")
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.createActions(names)
        self.createMenuBar()
        self.createToolBars(0)

        # TODO: finish toolbars and menubar functionality

    @property
    def cfg(self):
        return auxiliaryfunctions.read_config(self.config)

    @property
    def project_folder(self) -> str:
        return self.cfg.get("project_path", os.path.expanduser("~/Desktop"))

    @property
    def is_multianimal(self) -> bool:
        return bool(self.cfg.get("multianimalproject"))

    @property
    def all_bodyparts(self) -> List:
        if self.is_multianimal:
            return self.cfg.get("multianimalbodyparts")
        else:
            return self.cfg["bodyparts"]

    @property
    def all_individuals(self) -> List:
        if self.is_multianimal:
            return self.cfg.get("individuals")
        else:
            return [""]

    def update_shuffle(self, value):
        self.logger.info(f"Shuffle set to {value}")
        self.shuffle_value = value

    def window_set(self):
        self.setWindowTitle("DeepLabCut")
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("www.deeplabcut.org")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        self.setWindowIcon(QIcon(self.logo))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")

    def set_pic(self, name):
        pic_dir = os.path.dirname(os.path.realpath(name)) + os.path.sep
        file = pic_dir + "/assets/" + name
        pixmap = QPixmap(file)
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

        lbl_welcome1 = QLabel("Welcome to the DeepLabCut Project Manager GUI!")
        lbl_welcome1.setAlignment(Qt.AlignCenter)
        lbl_welcome2 = QLabel("To get started, please click on the 'File'")
        lbl_welcome2.setAlignment(Qt.AlignCenter)
        lbl_welcome3 = QLabel("tab to create or load an existing project.")
        lbl_welcome3.setAlignment(Qt.AlignCenter)

        layout.addWidget(lbl_welcome1)
        layout.addWidget(lbl_welcome2)
        layout.addWidget(lbl_welcome3)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def project_folder(self):
        return self.cfg.get("project_path", os.path.expanduser("~/Desktop"))

    def default_set(self):
        self.name_default = ""
        self.proj_default = ""
        self.exp_default = ""
        self.loc_default = str(Path.home())

    def createActions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")

        self.newAction.setIcon(QIcon("assets/icons/" + names[0]))
        self.newAction.setShortcut("Ctrl+N")

        self.newAction.triggered.connect(self._create_project)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(QIcon("assets/icons/" + names[1]))
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.triggered.connect(self._open_project)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(QIcon("assets/icons/" + names[2]))

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

    @QtCore.Slot()
    def _create_project(self):
        create_project = CreateProject(self)
        create_project.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if create_project.exec_() == QtWidgets.QDialog.Accepted:
            self.loaded = create_project.loaded
            self.config = create_project.config
            self.user_feedback = create_project.user_fbk

        if create_project.loaded:
            self.add_tabs()

    def _open_project(self):
        open_project = OpenProject(self)
        open_project.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if open_project.exec_() == QtWidgets.QDialog.Accepted:
            self.loaded = open_project.loaded
            self.config = open_project.config
            self.user_feedback = open_project.user_fbk

        if open_project.loaded:
            self.add_tabs()

    def load_config(self, config):
        self.config = config
        self.config_loaded.emit()
        print(f'Project "{self.cfg["Task"]}" successfully loaded.')

    def darkmode(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.welcome_page("Welcome2.png")

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.createActions(names)
        self.updateMenuBar()
        self.createToolBars(1)

    def lightmode(self):
        self.setStyleSheet("Fusion")
        self.welcome_page("Welcome.png")

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.createActions(names)
        self.createToolBars(1)
        self.updateMenuBar()

    def add_tabs(self):
        # Add all the other pages

        tabs = QtWidgets.QTabWidget()
        tabs.setContentsMargins(0, 20, 0, 0)
        extract_frames = ExtractFrames(root=self, parent=self, tab_heading="DeepLabCut - Step 2. Extract Frames")
        label_frames = LabelFrames(self, self.config)
        create_training_dataset = CreateTrainingDataset(self, self.config)
        train_network = TrainNetwork(self, self.config)
        evaluate_network = EvaluateNetwork(root=self, parent=self, tab_heading="DeepLabCut - Step 6. Evaluate Network")
        video_editor = VideoEditor(self, self.config)
        analyze_videos = AnalyzeVideos(root=self, parent=self, tab_heading="DeepLabCut - Step 7. Analyze Videos")
        create_videos = CreateVideos(root=self, parent=self, tab_heading="DeepLabCut - Optional Step. Create Videos")
        extract_outlier_frames = ExtractOutlierFrames(self, self.config)
        refine_labels = RefineLabels(self, self.config)

        tabs.addTab(extract_frames, "Extract frames")
        tabs.addTab(label_frames, "Label frames")
        tabs.addTab(create_training_dataset, "Create training dataset")
        tabs.addTab(train_network, "Train network")
        tabs.addTab(evaluate_network, "Evaluate network")
        tabs.addTab(video_editor, "Video editor")
        tabs.addTab(analyze_videos, "Analyze videos")
        tabs.addTab(create_videos, "Create videos")
        tabs.addTab(extract_outlier_frames, "Extract outlier frames")
        tabs.addTab(refine_labels, "Refine labels")

        self.setCentralWidget(tabs)
