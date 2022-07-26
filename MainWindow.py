import os
import logging
from pathlib import Path
from typing import List
import qdarkstyle

import deeplabcut
from deeplabcut import auxiliaryfunctions

from PySide2.QtWidgets import QAction, QMenu, QLabel, QVBoxLayout, QWidget, QMainWindow
from PySide2 import QtCore
from PySide2.QtGui import QPixmap, QIcon
from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import Qt
from components import (
    ShuffleSpinBox,
    TrainingSetSpinBox,
    _create_label_widget,
    _create_vertical_layout,
)

from create_project import CreateProject
from open_project import OpenProject
from extract_frames import ExtractFrames
from label_frames import LabelFrames
from create_training_dataset import CreateTrainingDataset
from train_network import TrainNetwork
from evaluate_network import EvaluateNetwork
from unsupervised_id_tracking import UnsupervizedIdTracking
from video_editor import VideoEditor
from analyze_videos import AnalyzeVideos
from create_videos import CreateVideos
from extract_outlier_frames import ExtractOutlierFrames
from refine_tracklets import RefineTracklets


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
        self.trainingset_index = 0
        self.videotype = "avi"
        self.files = set()

        self.default_set()

        self._generate_welcome_page()
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.createActions(names)
        self.createMenuBar()
        self.createToolBars(0)
        self.darkmode()

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

    @property
    def pose_cfg_path(self) -> str:
        try:
            return os.path.join(
                self.cfg["project_path"],
                auxiliaryfunctions.get_model_folder(
                    self.cfg["TrainingFraction"][int(self.trainingset_index)],
                    int(self.shuffle_value),
                    self.cfg,
                ),
                "train",
                "pose_cfg.yaml",
            )
        except FileNotFoundError:
            return Path(deeplabcut.__file__).parent / "pose_cfg.yaml"

    @property
    def inference_cfg_path(self) -> str:
        return os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                self.cfg["TrainingFraction"][int(self.trainingset_index)],
                int(self.shuffle_value),
                self.cfg,
            ),
            "test",
            "inference_cfg.yaml",
        )

    def update_cfg(self, text):
        self.root.config = text
        # Disable transformer_tracking tab if single animal
        self.unsupervised_id_tracking.setEnabled(True if self.is_multianimal else False)

    def update_shuffle(self, value):
        self.shuffle_value = value
        self.logger.info(f"Shuffle set to {self.shuffle_value}")

    def update_trainingset(self, value):
        self.trainingset_index = value
        self.logger.info(f"Trainingset index set to {self.trainingset_index}")

    def update_videotype(self, vtype):
        self.videotype = vtype
        self.logger.info(f"Videotype set to {self.videotype}")

    def update_files(self, files: set):
        self.files.update(files)
        self.logger.info(f"Videos selected to analyze:\n{self.files}")

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

    def _generate_welcome_page(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.layout.setSpacing(30)

        self.layout.addWidget(
            _create_label_widget(
                "Welcome to the DeepLabCut Project Manager GUI!",
                "font:bold; font-size:18px;",
                margins=(0, 30, 0, 0),
            )
        )

        logo = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "logo_transparent.png"
        )

        image_widget = QtWidgets.QLabel(self)
        image_widget.setContentsMargins(0, 0, 0, 0)
        image_widget.setFixedHeight(400)
        image_widget.setFixedWidth(400)
        pixmap = QtGui.QPixmap(logo)
        image_widget.setPixmap(pixmap.scaledToHeight(400))
        self.layout.addWidget(image_widget)

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.create_project_button = QtWidgets.QPushButton("Create New Project")
        self.create_project_button.setFixedWidth(200)
        self.create_project_button.clicked.connect(self._create_project)

        self.load_project_button = QtWidgets.QPushButton("Load Project")
        self.load_project_button.setFixedWidth(200)
        self.load_project_button.clicked.connect(self._open_project)

        self.layout_buttons.addWidget(self.create_project_button)
        self.layout_buttons.addWidget(self.load_project_button)

        self.layout.addLayout(self.layout_buttons)

        widget = QWidget()
        widget.setLayout(self.layout)
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
        open_project.load_config()
        if not open_project.config:
            return
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

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.createActions(names)
        self.updateMenuBar()
        self.createToolBars(1)

    def lightmode(self):
        stylefile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "style.qss"
        )
        with open(stylefile, "r") as f:
            self.setStyleSheet(f.read())

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.createActions(names)
        self.createToolBars(1)
        self.updateMenuBar()

    def add_tabs(self):
        # Add all the other pages

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.extract_frames = ExtractFrames(
            root=self, parent=self, h1_description="DeepLabCut - Step 2. Extract Frames"
        )
        self.label_frames = LabelFrames(
            root=self, parent=self, h1_description="DeepLabCut - Step 3. Label Frames"
        )
        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=self,
            h1_description="DeepLabCut - Step 4. Create training dataset",
        )
        self.train_network = TrainNetwork(
            root=self, parent=self, h1_description="DeepLabCut - Step 5. Train network"
        )
        self.evaluate_network = EvaluateNetwork(
            root=self,
            parent=self,
            h1_description="DeepLabCut - Step 6. Evaluate Network",
        )
        self.analyze_videos = AnalyzeVideos(
            root=self, parent=self, h1_description="DeepLabCut - Step 7. Analyze Videos"
        )
        self.unsupervised_id_tracking = UnsupervizedIdTracking(
            root=self,
            parent=self,
            h1_description="DeepLabCut - Optional Unsupervised ID Tracking with Transformer",
        )
        self.create_videos = CreateVideos(
            root=self,
            parent=self,
            h1_description="DeepLabCut - Optional Step. Create Videos",
        )
        self.extract_outlier_frames = ExtractOutlierFrames(
            root=self,
            parent=self,
            h1_description="DeepLabCut - Step 8. Extract outlier frame",
        )
        self.refine_tracklets = RefineTracklets(
            root=self, parent=self, h1_description="DeepLabCut - Step 9. Refine labels"
        )
        self.video_editor = VideoEditor(
            root=self, parent=self, h1_description="DeepLabCut - Optional Video Editor"
        )

        self.tab_widget.addTab(self.extract_frames, "Extract frames")
        self.tab_widget.addTab(self.label_frames, "Label frames")
        self.tab_widget.addTab(self.create_training_dataset, "Create training dataset")
        self.tab_widget.addTab(self.train_network, "Train network")
        self.tab_widget.addTab(self.evaluate_network, "Evaluate network")
        self.tab_widget.addTab(self.analyze_videos, "Analyze videos")
        self.tab_widget.addTab(
            self.unsupervised_id_tracking, "Unsupervised ID Tracking (*)"
        )
        if not self.is_multianimal:
            self.unsupervised_id_tracking.setEnabled(False)
        self.tab_widget.addTab(self.create_videos, "Create videos")
        self.tab_widget.addTab(
            self.extract_outlier_frames, "Extract outlier frames (*)"
        )
        self.tab_widget.addTab(self.refine_tracklets, "Refine tracklets (*)")
        self.tab_widget.addTab(self.video_editor, "Video editor (*)")

        self.setCentralWidget(self.tab_widget)

        self.tab_widget.currentChanged.connect(self.refresh_active_tab)

    def refresh_active_tab(self):
        active_tab = self.tab_widget.currentWidget()
        tab_label = self.tab_widget.tabText(self.tab_widget.currentIndex())

        widget_to_attribute_map = {
            QtWidgets.QSpinBox: "setValue",
            ShuffleSpinBox: "setValue",
            TrainingSetSpinBox: "setValue",
            QtWidgets.QLineEdit: "setText",
        }

        def _attempt_attribute_update(widget_name, updated_value):
            try:
                widget = getattr(active_tab, widget_name)
                method = getattr(widget, widget_to_attribute_map[type(widget)])
                self.logger.debug(
                    f"Setting {widget_name}={updated_value} in tab '{tab_label}'"
                )
                method(updated_value)
            except AttributeError:
                self.logger.debug(
                    f"Tab '{tab_label}' has no attribute named {widget_name}. Skipping..."
                )

        def _attempt_video_widget_update(videotype, selected_videos):
            # TODO: NOT WORKING
            try:
                video_widget = active_tab.video_selection_widget
                self.logger.debug(
                    f"Setting videotype={videotype} and videos={active_tab.video_selection_widget.files} in tab '{tab_label}'"
                )
                video_widget.videotype_widget.setCurrentText(videotype)
                video_widget._update_video_selection(selected_videos)
            except AttributeError:
                self.logger.debug(
                    f"Tab '{tab_label}' has no attribute video_selection_widget. Skipping..."
                )

        _attempt_attribute_update("shuffle", self.shuffle_value)
        _attempt_attribute_update("trainingset", self.trainingset_index)
        _attempt_attribute_update("cfg_line", self.config)
        # _attempt_video_widget_update(self.videotype, self.files)

        # Update single/multi animal menus
        # TODO

