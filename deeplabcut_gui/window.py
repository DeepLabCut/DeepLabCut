import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import List
import qdarkstyle

import deeplabcut
from deeplabcut import auxiliaryfunctions, VERSION
from deeplabcut_gui import BASE_DIR, utils, components

from PySide2.QtWidgets import QAction, QMenu, QWidget, QMainWindow
from PySide2 import QtCore
from PySide2.QtGui import QIcon
from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import Qt

from tabs import *
from widgets import StreamReceiver, StreamWriter


def _check_for_updates():
    is_latest, latest_version = utils.is_latest_deeplabcut_version()
    if not is_latest:
        msg = QtWidgets.QMessageBox(
            text=f"DeepLabCut {latest_version} available",
        )
        msg.setIcon(QtWidgets.QMessageBox.Information)
        update_btn = msg.addButton('Update', msg.AcceptRole)
        msg.setDefaultButton(update_btn)
        _ = msg.addButton('Skip', msg.RejectRole)
        msg.exec_()
        if msg.clickedButton() is update_btn:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-U', 'deeplabcut']
            )
    else:
        msg = QtWidgets.QMessageBox(
            text=f"DeepLabCut is up-to-date",
        )
        msg.exec_()


class MainWindow(QMainWindow):

    config_loaded = QtCore.Signal()
    video_type_ = QtCore.Signal(str)
    video_files_ = QtCore.Signal(set)

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.app = app
        desktop = QtWidgets.QDesktopWidget().screenGeometry(0)
        self.screen_width = desktop.width()
        self.screen_height = desktop.height()

        self.logger = logging.getLogger("GUI")

        self.config = None
        self.loaded = False
        self.user_feedback = False

        self.shuffle_value = 1
        self.trainingset_index = 0
        self.videotype = "mp4"
        self.files = set()

        self.default_set()

        self._generate_welcome_page()
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.create_actions(names)
        self.createMenuBar()
        self.createToolBars(0)

        # Thread-safe Stdout redirector
        self.status_bar = self.statusBar()
        self.status_bar.setObjectName('Status Bar')
        self.writer = StreamWriter()
        sys.stdout = self.writer
        self.receiver = StreamReceiver(self.writer.queue)
        self.receiver.new_text.connect(self.print_to_status_bar)

    def print_to_status_bar(self, text):
        self.status_bar.showMessage(text)
        self.status_bar.repaint()

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
            return str(Path(deeplabcut.__file__).parent / "pose_cfg.yaml")

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
        self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

    def update_shuffle(self, value):
        self.shuffle_value = value
        self.logger.info(f"Shuffle set to {self.shuffle_value}")

    @property
    def video_type(self):
        return self.videotype

    @video_type.setter
    def video_type(self, ext):
        self.videotype = ext
        self.video_type_.emit(ext)
        self.logger.info(f"Video type set to {self.video_type}")

    @property
    def video_files(self):
        return self.files

    @video_files.setter
    def video_files(self, video_files):
        self.files = set(video_files)
        self.video_files_.emit(self.files)
        self.logger.info(f"Videos selected to analyze:\n{self.files}")

    def window_set(self):
        self.setWindowTitle("DeepLabCut")
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("www.deeplabcut.org")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        icon = os.path.join(BASE_DIR, 'assets', 'logo.png')
        self.setWindowIcon(QIcon(icon))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")

    def _generate_welcome_page(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.layout.setSpacing(30)

        title = components._create_label_widget(
            f"Welcome to the DeepLabCut Project Manager GUI {VERSION}!",
            "font:bold; font-size:18px;",
            margins=(0, 30, 0, 0),
        )
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        image_widget = QtWidgets.QLabel(self)
        image_widget.setAlignment(Qt.AlignCenter)
        image_widget.setContentsMargins(0, 0, 0, 0)
        logo = os.path.join(BASE_DIR, "assets", "logo_transparent.png")
        pixmap = QtGui.QPixmap(logo)
        image_widget.setPixmap(
            pixmap.scaledToHeight(400, QtCore.Qt.SmoothTransformation)
        )
        self.layout.addWidget(image_widget)

        description = "DeepLabCut™ is an open source tool for markerless pose estimation of user-defined body parts with deep learning.\nA.  and M.W.  Mathis Labs | http://www.deeplabcut.org\n\n To get started,  create a new project or load an existing one."
        label = components._create_label_widget(
            description,
            "font-size:12px; text-align: center;",
            margins=(0, 0, 0, 0),
        )
        label.setMinimumWidth(400)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)

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

    def create_actions(self, names):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project...")

        self.newAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[0]))
        )
        self.newAction.setShortcut("Ctrl+N")

        self.newAction.triggered.connect(self._create_project)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[1]))
        )
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.triggered.connect(self._open_project)

        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)

        self.lightmodeAction = QAction("&Light theme", self)
        self.lightmodeAction.triggered.connect(self.lightmode)
        self.darkmodeAction = QAction("&Dark theme", self)
        self.darkmodeAction.triggered.connect(self.darkmode)

        self.helpAction = QAction("&Help", self)
        self.helpAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[2]))
        )

        self.aboutAction = QAction("&Learn DLC", self)
        self.check_updates = QAction("&Check for Updates...", self)
        self.check_updates.triggered.connect(_check_for_updates)

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
        helpMenu.addAction(self.check_updates)
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
        dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
        self.app.setStyleSheet(dark_stylesheet)

        names = ["new_project2.png", "open2.png", "help2.png"]
        self.remove_action()
        self.create_actions(names)
        self.updateMenuBar()
        self.createToolBars(1)

    def lightmode(self):
        from qdarkstyle.light.palette import LightPalette

        style = qdarkstyle.load_stylesheet(palette=LightPalette)
        self.app.setStyleSheet(style)

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.create_actions(names)
        self.createToolBars(1)
        self.updateMenuBar()

    def add_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.manage_project = ManageProject(
            root=self, parent=None, h1_description="DeepLabCut - Step 1. Manage Project"
        )
        self.extract_frames = ExtractFrames(
            root=self, parent=None, h1_description="DeepLabCut - Step 2. Extract Frames"
        )
        self.label_frames = LabelFrames(
            root=self, parent=None, h1_description="DeepLabCut - Step 3. Label Frames"
        )
        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Step 4. Create training dataset",
        )
        self.train_network = TrainNetwork(
            root=self, parent=None, h1_description="DeepLabCut - Step 5. Train network"
        )
        self.evaluate_network = EvaluateNetwork(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Step 6. Evaluate Network",
        )
        self.analyze_videos = AnalyzeVideos(
            root=self, parent=None, h1_description="DeepLabCut - Step 7. Analyze Videos"
        )
        self.unsupervised_id_tracking = UnsupervizedIdTracking(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Optional Unsupervised ID Tracking with Transformer",
        )
        self.create_videos = CreateVideos(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Optional Step. Create Videos",
        )
        self.extract_outlier_frames = ExtractOutlierFrames(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Step 8. Extract outlier frame",
        )
        self.refine_tracklets = RefineTracklets(
            root=self, parent=None, h1_description="DeepLabCut - Step 9. Refine labels"
        )
        self.video_editor = VideoEditor(
            root=self, parent=None, h1_description="DeepLabCut - Optional Video Editor"
        )

        self.tab_widget.addTab(self.manage_project, "Manage project")
        self.tab_widget.addTab(self.extract_frames, "Extract frames")
        self.tab_widget.addTab(self.label_frames, "Label frames")
        self.tab_widget.addTab(self.create_training_dataset, "Create training dataset")
        self.tab_widget.addTab(self.train_network, "Train network")
        self.tab_widget.addTab(self.evaluate_network, "Evaluate network")
        self.tab_widget.addTab(self.analyze_videos, "Analyze videos")
        self.tab_widget.addTab(
            self.unsupervised_id_tracking, "Unsupervised ID Tracking (*)"
        )
        self.tab_widget.addTab(self.create_videos, "Create videos")
        self.tab_widget.addTab(
            self.extract_outlier_frames, "Extract outlier frames (*)"
        )
        self.tab_widget.addTab(self.refine_tracklets, "Refine tracklets (*)")
        self.tab_widget.addTab(self.video_editor, "Video editor (*)")

        if not self.is_multianimal:
            self.refine_tracklets.setEnabled(False)
        self.unsupervised_id_tracking.setEnabled(self.is_transreid_available())

        self.setCentralWidget(self.tab_widget)

        self.tab_widget.currentChanged.connect(self.refresh_active_tab)

    def refresh_active_tab(self):
        active_tab = self.tab_widget.currentWidget()
        tab_label = self.tab_widget.tabText(self.tab_widget.currentIndex())

        widget_to_attribute_map = {
            QtWidgets.QSpinBox: "setValue",
            components.ShuffleSpinBox: "setValue",
            components.TrainingSetSpinBox: "setValue",
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

        _attempt_attribute_update("shuffle", self.shuffle_value)
        _attempt_attribute_update("cfg_line", self.config)

        # Update single/multi animal menus
        # TODO

    def is_transreid_available(self):
        if self.is_multianimal:
            try:
                from deeplabcut.pose_tracking_pytorch import transformer_reID
                return True
            except ModuleNotFoundError:
                return False
        else:
            return False

    def closeEvent(self, event):
        print('Exiting...')
        answer = QtWidgets.QMessageBox.question(self, 'Quit',
                                                'Are you sure you want to quit?',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                                                QtWidgets.QMessageBox.Cancel)
        if answer == QtWidgets.QMessageBox.Yes:
            self.receiver.terminate()
            event.accept()
        else:
            event.ignore()
            print('')
