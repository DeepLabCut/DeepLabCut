import os
import qdarkstyle

from deeplabcut import auxiliaryfunctions

from PyQt5.QtWidgets import QAction, QMenu, QLabel, QVBoxLayout, QWidget
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

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


class MainWindow(QtWidgets.QMainWindow):
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

        self.welcome_page("Welcome.png")
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.createActions(names)
        self.createMenuBar()
        self.createToolBars(0)

        # TODO: finish toolbars and menubar functionality

    def window_set(self):
        self.setWindowTitle("DeepLabCut")
        self.setMinimumSize(1500, 750)
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("www.deeplabcut.org")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/pictures/logo.png"
        self.setWindowIcon(QIcon(self.logo))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")

    def set_pic(self, name):
        pic_dir = os.path.dirname(os.path.realpath(name)) + os.path.sep
        file = pic_dir + "/pictures/" + name
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
        lbl_welcome3 = QLabel("tab to cteate or load an existing project.")
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
        self.loc_default = "C:/"

    def createActions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")

        self.newAction.setIcon(QIcon("icons/" + names[0]))
        self.newAction.setShortcut("Ctrl+N")

        self.newAction.triggered.connect(self._create)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(QIcon("icons/" + names[1]))
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.triggered.connect(self._open)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(QIcon("icons/" + names[2]))

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

    @QtCore.pyqtSlot()
    def _create(self):
        create_p = CreateProject(self)
        create_p.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if create_p.exec_() == QtWidgets.QDialog.Accepted:
            self.loaded = create_p.loaded
            self.cfg = create_p.cfg
            self.user_feedback = create_p.user_fbk

        if create_p.loaded:
            self.add_tabs()

    def _open(self):
        open_p = OpenProject(self)
        open_p.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if open_p.exec_() == QtWidgets.QDialog.Accepted:
            self.loaded = open_p.loaded
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
        extract_frames = ExtractFrames(self, self.cfg)
        label_frames = LabelFrames(self, self.cfg)
        create_training_dataset = CreateTrainingDataset(self, self.cfg)
        train_network = TrainNetwork(self, self.cfg)
        evaluate_network = EvaluateNetwork(self, self.cfg)
        video_editor = VideoEditor(self, self.cfg)
        analyze_videos = AnalyzeVideos(self, self.cfg)
        create_videos = CreateVideos(self, self.cfg)
        extract_outlier_frames = ExtractOutlierFrames(self, self.cfg)
        refine_labels = RefineLabels(self, self.cfg)

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
