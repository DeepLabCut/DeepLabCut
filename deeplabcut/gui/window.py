#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
import logging
import subprocess
import sys
from functools import cached_property
from pathlib import Path
from typing import List
from urllib.error import URLError
import qdarkstyle

import deeplabcut
from deeplabcut import auxiliaryfunctions, VERSION, compat
from deeplabcut.core.engine import Engine
from deeplabcut.gui import BASE_DIR, components, utils
from deeplabcut.gui.tabs import *
from deeplabcut.gui.widgets import StreamReceiver, StreamWriter
from deeplabcut.utils.multiprocessing import call_with_timeout
from napari_deeplabcut import misc
from PySide6.QtWidgets import (
    QMessageBox,
    QMenu,
    QWidget,
    QMainWindow,
    QComboBox,
    QLabel,
    QSizePolicy,
)
from PySide6 import QtCore
from PySide6.QtGui import QIcon, QAction, QPixmap
from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt, QTimer


def _check_for_updates(silent=True):
    try:
        is_latest, latest_version = call_with_timeout(
            utils.is_latest_deeplabcut_version, 5
        )
        is_latest_plugin, latest_plugin_version = call_with_timeout(
            misc.is_latest_version, 5
        )
    except (URLError, TimeoutError):  # Handle internet connectivity issues
        is_latest = is_latest_plugin = True

    if is_latest and is_latest_plugin:
        if not silent:
            msg = QtWidgets.QMessageBox(
                text=f"DeepLabCut is up-to-date",
            )
            msg.exec_()
    else:
        if not is_latest and is_latest_plugin:
            text = f"DeepLabCut {latest_version} available"
            command = "pip", "install", "-U", "deeplabcut"
        elif not is_latest_plugin and is_latest:
            text = f"DeepLabCut labeling plugin {latest_plugin_version} available"
            command = "pip", "install", "-U", "napari-deeplabcut"
        else:
            text = f"DeepLabCut {latest_version}\nand labeling plugin {latest_plugin_version} available"
            command = "pip", "install", "-U", "deeplabcut", "napari-deeplabcut"

        msg = QtWidgets.QMessageBox(
            text=text,
        )
        msg.setIcon(QtWidgets.QMessageBox.Information)
        update_btn = msg.addButton("Update", QtWidgets.QMessageBox.AcceptRole)
        msg.setDefaultButton(update_btn)
        _ = msg.addButton("Skip", QtWidgets.QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() is update_btn:
            subprocess.check_call([sys.executable, "-m", *command])


class MainWindow(QMainWindow):
    config_loaded = QtCore.Signal()
    video_type_ = QtCore.Signal(str)
    video_files_ = QtCore.Signal(set)
    engine_change = QtCore.Signal(Engine)
    shuffle_change = QtCore.Signal(int)
    shuffle_created = QtCore.Signal(int)

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.app = app
        screen_size = app.screens()[0].size()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()

        self.logger = logging.getLogger("GUI")

        self.config = None
        self.loaded = False

        self.shuffle_value = 1
        self.trainingset_index = 0
        self.videotype = "mp4"
        self.files = set()

        self._engine = Engine.PYTORCH

        self.default_set()

        self._generate_welcome_page()
        self.window_set()
        self.default_set()

        names = ["new_project.png", "open.png", "help.png"]
        self.create_actions(names)
        self.create_menu_bar()
        self.load_settings()
        self._toolbar = None
        self.create_toolbar()

        # Thread-safe Stdout redirector
        self.writer = StreamWriter()
        sys.stdout = self.writer
        self.receiver = StreamReceiver(self.writer.queue)
        self.receiver.new_text.connect(self.print_to_status_bar)

        # create logger to also log to the console
        logging.basicConfig()
        logging.getLogger("console").setLevel(logging.INFO)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMaximum(0)
        self._progress_bar.hide()
        self.status_bar.addPermanentWidget(self._progress_bar)

    def print_to_status_bar(self, text):
        self.status_bar.showMessage(text)
        self.status_bar.repaint()
        logging.getLogger("console").info(text)

    @property
    def toolbar(self):
        if self._toolbar is None:
            self._toolbar = self.addToolBar("File")
        return self._toolbar

    @cached_property
    def settings(self):
        return QtCore.QSettings()

    def load_settings(self):
        filenames = self.settings.value("recent_files") or []
        for filename in filenames:
            self.add_recent_filename(filename)

    def save_settings(self):
        recent_files = []
        for action in self.recentfiles_menu.actions()[::-1]:
            recent_files.append(action.text())
        self.settings.setValue("recent_files", recent_files)

    def add_recent_filename(self, filename):
        actions = self.recentfiles_menu.actions()
        filenames = [action.text() for action in actions]
        if filename in filenames:
            return
        action = QAction(filename, self)
        before_action = actions[0] if actions else None
        self.recentfiles_menu.insertAction(before_action, action)

    @property
    def cfg(self):
        try:
            cfg = auxiliaryfunctions.read_config(self.config)
        except TypeError:
            cfg = {}
        return cfg

    @property
    def engine(self) -> Engine:
        return self._engine

    @engine.setter
    def engine(self, e: Engine) -> None:
        if self._engine == e:
            return

        if e == e.TF:
            try:
                import tensorflow
            except ModuleNotFoundError as err:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Cannot use the TensorFlow engine.")
                msg.setInformativeText(
                    f"Error `{err}`\nCannot use the TensorFlow engine as TensorFlow "
                    "is not installed. To use it, install TensorFlow with\n"
                    "    Windows/Linux:\n"
                    "        pip install 'deeplabcut[tf]'\n"
                    "    Apple Silicon:\n"
                    "        pip install 'deeplabcut[apple_mchips]'\n\n"
                    "Please switch back to the PyTorch engine to use DeepLabCut, or install TensorFlow."
                )

                msg.setWindowTitle("Info")
                msg.setMinimumWidth(900)
                logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
                logo = logo_dir + "/assets/logo.png"
                msg.setWindowIcon(QIcon(logo))
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

        self._engine = e
        self.engine_change.emit(e)

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
            return str(
                compat.return_train_network_path(
                    self.config,
                    shuffle=int(self.shuffle_value),
                    trainingsetindex=int(self.trainingset_index),
                    modelprefix="",
                )[0]
            )
        except FileNotFoundError:
            return str(Path(deeplabcut.__file__).parent / "pose_cfg.yaml")

    @property
    def models_folder(self) -> str:
        try:
            return str(
                compat.return_train_network_path(
                    self.config,
                    shuffle=int(self.shuffle_value),
                    trainingsetindex=int(self.trainingset_index),
                    modelprefix="",
                )[2]
            )
        except FileNotFoundError:
            return self.project_folder()

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
        self.shuffle_change.emit(value)
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

    def add_video_files(self, new_video_files):
        """
        Add new video files to the existing set of files. This method ensures no duplicates are added.
        Emits a signal to notify about the updated set of files.
        """
        new_video_files = set(new_video_files)
        self.files.update(new_video_files) # Add new items to the existing set
        self.video_files_.emit(self.files) # Emit the updated set of files
        self.logger.info(f"Videos added to analyze:\n{new_video_files}\nCurrent video files:\n{self.files}")

    def clear_video_files(self):
        """
        Clear all video files from the existing set. Emits a signal to notify the change.
        """
        self.files.clear()  # Reset the set to be empty
        self.video_files_.emit(self.files)  # Emit the empty set
        self.logger.info("All video files have been cleared.")

    def window_set(self):
        self.setWindowTitle("DeepLabCut")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        icon = os.path.join(BASE_DIR, "assets", "logo.png")
        self.setWindowIcon(QIcon(icon))

        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("Status Bar")
        self.status_bar.showMessage("www.deeplabcut.org")

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

        description = "DeepLabCut™ is an open source tool for markerless pose estimation of user-defined body parts with deep learning.\nA.  and M.W.  Mathis Labs | http://www.deeplabcut.org\n\n To get started,  create a new project, load an existing one, or try one of our pretrained models from the Model Zoo."
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

        self.run_superanimal_button = QtWidgets.QPushButton("Model Zoo")
        self.run_superanimal_button.setFixedWidth(200)
        self.run_superanimal_button.clicked.connect(self._goto_superanimal)

        self.layout_buttons.addWidget(self.create_project_button)
        self.layout_buttons.addWidget(self.load_project_button)
        self.layout_buttons.addWidget(self.run_superanimal_button)

        self.layout.addLayout(self.layout_buttons)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        QTimer.singleShot(1000, lambda: _check_for_updates(silent=True))

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
        self.newAction.setStatusTip("Create a new project...")

        self.newAction.triggered.connect(self._create_project)

        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.openAction.setIcon(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", names[1]))
        )
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setStatusTip("Open a project...")
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
        self.helpAction.setStatusTip("Ask for help...")
        self.helpAction.triggered.connect(self._ask_for_help)

        self.aboutAction = QAction("&Learn DLC", self)
        self.aboutAction.triggered.connect(self._learn_dlc)

        self.check_updates = QAction("&Check for Updates...", self)
        self.check_updates.triggered.connect(lambda: _check_for_updates(silent=False))

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        menu_bar.addMenu(self.file_menu)

        self.file_menu.addAction(self.newAction)
        self.file_menu.addAction(self.openAction)

        self.recentfiles_menu = self.file_menu.addMenu("Open Recent")
        self.recentfiles_menu.triggered.connect(
            lambda a: self._update_project_state(a.text(), True)
        )
        self.file_menu.addAction(self.saveAction)
        self.file_menu.addAction(self.exitAction)

        # View menu
        view_menu = QMenu("&View", self)
        mode = view_menu.addMenu("Appearance")
        menu_bar.addMenu(view_menu)
        mode.addAction(self.lightmodeAction)
        mode.addAction(self.darkmodeAction)

        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)
        help_menu.addAction(self.helpAction)
        help_menu.adjustSize()
        help_menu.addAction(self.check_updates)
        help_menu.addAction(self.aboutAction)

    def update_menu_bar(self):
        self.file_menu.removeAction(self.newAction)
        self.file_menu.removeAction(self.openAction)

    def create_toolbar(self):
        self.toolbar.clear()
        self.toolbar.addAction(self.newAction)
        self.toolbar.addAction(self.openAction)
        self.toolbar.addAction(self.helpAction)

        size_policy = QSizePolicy()  # QtWidgets.QSizePolicy.Policy.Expanding
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        spacer = QLabel()
        spacer.setSizePolicy(size_policy)
        spacer.setStyleSheet("background: transparent;")

        engine_label = QLabel()
        engine_label.autoFillBackground()
        engine_label.setText("Engine")
        engine_label.setStyleSheet("background: transparent;")

        engine_icon = QLabel()
        engine_icon.setStyleSheet("background: transparent;")

        def _update_icon(engine: str):
            pixmap = QPixmap(f"deeplabcut/gui/media/dlc-{engine}.png")
            engine_icon.setPixmap(
                pixmap.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio)
            )

        _update_icon("pt" if self.engine == Engine.PYTORCH else "tf")

        engines = [engine for engine in Engine]

        def _update_engine(index: int) -> None:
            self.logger.info(f"Changed engine to {engines[index]}")
            self.engine = engines[index]
            _update_icon("pt" if self.engine == Engine.PYTORCH else "tf")

        change_engine_widget = QComboBox()
        change_engine_widget.addItems([e.aliases[0] for e in engines])
        change_engine_widget.setFixedWidth(180)
        change_engine_widget.currentIndexChanged.connect(_update_engine)
        change_engine_widget.setCurrentIndex(engines.index(self.engine))

        self.toolbar.addWidget(spacer)
        self.toolbar.addWidget(engine_icon)
        self.toolbar.addWidget(engine_label)
        self.toolbar.addWidget(change_engine_widget)

    def remove_action(self):
        self.toolbar.removeAction(self.newAction)
        self.toolbar.removeAction(self.openAction)
        self.toolbar.removeAction(self.helpAction)

    def _update_project_state(self, config, loaded):
        self.config = config
        self.loaded = loaded
        if loaded:
            self.add_recent_filename(self.config)
            self.add_tabs()

    def _ask_for_help(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ask for help")
        dlg.setText(
            """Ask our community for help on <a href='https://forum.image.sc/tag/deeplabcut'>the forum</a>!"""
        )
        _ = dlg.exec()

    def _learn_dlc(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Learn DLC")
        dlg.setText(
            """Learn DLC with <a href='https://deeplabcut.github.io/DeepLabCut/docs/UseOverviewGuide.html'>our docs and how-to guides</a>!"""
        )
        _ = dlg.exec()

    def _create_project(self):
        dlg = ProjectCreator(self)
        dlg.show()

    def _open_project(self):
        open_project = OpenProject(self)
        open_project.load_config()
        if not open_project.config:
            return

        open_project.loaded = True
        self._update_project_state(
            open_project.config,
            open_project.loaded,
        )

    def _goto_superanimal(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.modelzoo = ModelZoo(
            root=self, parent=None, h1_description="DeepLabCut - Model Zoo"
        )
        self.tab_widget.addTab(self.modelzoo, "Model Zoo")
        self.setCentralWidget(self.tab_widget)

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
        self.update_menu_bar()
        self.create_toolbar()

    def lightmode(self):
        from qdarkstyle.light.palette import LightPalette

        style = qdarkstyle.load_stylesheet(palette=LightPalette)
        self.app.setStyleSheet(style)

        names = ["new_project.png", "open.png", "help.png"]
        self.remove_action()
        self.create_actions(names)
        self.create_toolbar()
        self.update_menu_bar()

    def add_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setContentsMargins(0, 20, 0, 0)
        self.manage_project = ManageProject(
            root=self, parent=None, h1_description="DeepLabCut - Manage Project"
        )
        self.extract_frames = ExtractFrames(
            root=self, parent=None, h1_description="DeepLabCut - Extract Frames"
        )
        self.label_frames = LabelFrames(
            root=self, parent=None, h1_description="DeepLabCut - Label Frames"
        )
        self.create_training_dataset = CreateTrainingDataset(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Step 4. Create training dataset",
        )
        self.train_network = TrainNetwork(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Train network",
        )
        self.evaluate_network = EvaluateNetwork(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Evaluate Network",
        )
        self.analyze_videos = AnalyzeVideos(
            root=self, parent=None, h1_description="DeepLabCut - Analyze Videos"
        )
        self.unsupervised_id_tracking = UnsupervizedIdTracking(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Optional Unsupervised ID Tracking with Transformer",
        )
        self.create_videos = CreateVideos(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Create Videos",
        )
        self.extract_outlier_frames = ExtractOutlierFrames(
            root=self,
            parent=None,
            h1_description="DeepLabCut - Step 8. Extract outlier frames",
        )
        self.refine_tracklets = RefineTracklets(
            root=self, parent=None, h1_description="DeepLabCut - Refine labels"
        )
        self.modelzoo = ModelZoo(
            root=self, parent=None, h1_description="DeepLabCut - Model Zoo"
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
        self.tab_widget.addTab(self.modelzoo, "Model Zoo")
        self.tab_widget.addTab(self.video_editor, "Video editor (*)")

        if not self.is_multianimal:
            self.tab_widget.removeTab(
                self.tab_widget.indexOf(self.unsupervised_id_tracking)
            )
            self.tab_widget.removeTab(self.tab_widget.indexOf(self.refine_tracklets))

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
                pass

        _attempt_attribute_update("shuffle", self.shuffle_value)
        _attempt_attribute_update("cfg_line", self.config)

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
        print("Exiting...")
        answer = QtWidgets.QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.receiver.terminate()
            event.accept()
            self.save_settings()
        else:
            event.ignore()
            print("")
