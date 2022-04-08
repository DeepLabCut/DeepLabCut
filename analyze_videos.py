import logging

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from PySide2.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from widgets import ConfigEditor


def _add_label_widget(
    text: str, layout: QtWidgets.QLayout, margins: tuple = (20, 50, 0, 0)
) -> None:

    label = QtWidgets.QLabel(text)
    label.setContentsMargins(*margins)
    label.setStyleSheet("font:bold")

    layout.addWidget(label)


def _create_horizontal_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QHBoxLayout():

    layout = QtWidgets.QHBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout

def _create_vertical_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QHBoxLayout():

    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


class AnalyzeVideos(QWidget):
    def __init__(self, parent, cfg):
        super(AnalyzeVideos, self).__init__(parent)

        self.logger = logging.getLogger("GUI")

        self.filelist = set()
        self.all_bodyparts = []
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)

        if self.cfg["multianimalproject"]:
            self.all_bodyparts = self.cfg["multianimalbodyparts"]
            self.tracker_method = self.cfg.get("default_track_method", "ellipse")
        else:
            self.all_bodyparts = self.cfg["bodyparts"]

        # NOTE: Several functions can support selected bodyparts instead of all.
        #       Do I expose these? They're not exposed in the current GUI as far
        #       as I can see. (only plot_trajectories)
        self.backend_variables = {
            "save_as_csv": False,
            "dynamic_cropping": False,
            "input_video_type": "avi",
            "show_figures": True,
            "calibrate_assembly": False,
            "assemble_with_ID_only": False,
            "overwrite_tracks": False,
            "filter_data": False,
            "plot_trajectories": False,
            "plot_trajectories": False,
            "bodyparts_to_use": self.all_bodyparts,
            "create_video_all_detections": False,
            "robustnframes": False, #Use ffprobe
        }

        self.outer_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.outer_layout)

        self.set_page()

    def set_page(self):
        workflow_title = QtWidgets.QLabel("DeepLabCut - Step 7. Analyze Videos ....")
        workflow_title.setStyleSheet("font:bold; font-size:18px;")
        workflow_title.setContentsMargins(20, 20, 0, 10)

        self.outer_layout.addWidget(workflow_title)

        layout_config = _create_horizontal_layout()
        self._generate_config_layout(layout_config)
        self.outer_layout.addLayout(layout_config)

        _add_label_widget("Video Selection", self.outer_layout)
        self.layout_video_analysis = _create_horizontal_layout()
        self._generate_layout_video_analysis(self.layout_video_analysis)
        self.outer_layout.addLayout(self.layout_video_analysis)

        _add_label_widget("Analysis Attributes", self.outer_layout)
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)
        self.outer_layout.addLayout(self.layout_attributes)

        # Single / Multi animal Only Layouts
        self.layout_single_animal = _create_horizontal_layout()

        self.layout_multi_animal = _create_vertical_layout()

        if self.cfg["multianimalproject"]:
            # multianimal only:
            #   "Use ffprobe to read video metadata (slow but robust)",
            #   "Create video for checking detections",
            #   "Specify the Tracker Method (you can try each)"
            #   "Overwrite tracking files (set to yes if you edit inference parameters)",
            #   "Calibrate animal assembly?",
            #   "Assemble with identity only?",
            #   "Prioritize past connections over a window of size:")
            _add_label_widget("Multi-animal settings", self.outer_layout)
            self._generate_layout_multianimal_only_options(self.layout_multi_animal)
            self.outer_layout.addLayout(self.layout_multi_animal)
        else:
            # Single animal only
            #   dynamically crop bdpts
            _add_label_widget("Single-animal settings", self.outer_layout)
            self._generate_layout_single_animal(self.layout_single_animal)
            self.outer_layout.addLayout(self.layout_single_animal)

        _add_label_widget("Data Processing", self.outer_layout)
        self.layout_data_processing = _create_horizontal_layout()
        self._generate_layout_data_processing(self.layout_data_processing)
        self.outer_layout.addLayout(self.layout_data_processing)

        _add_label_widget("Visualization", self.outer_layout)
        self.layout_visualization = _create_horizontal_layout()
        self._generate_layout_visualization(self.layout_visualization)
        self.outer_layout.addLayout(self.layout_visualization)

        self.analyze_videos_btn = QtWidgets.QPushButton("Analyze Videos")
        self.analyze_videos_btn.clicked.connect(self.analyze_videos)

        self.edit_config_file_btn = QtWidgets.QPushButton("Edit config.yaml")
        self.edit_config_file_btn.clicked.connect(self.edit_config_file)

        self.outer_layout.addWidget(self.analyze_videos_btn, alignment=Qt.AlignRight)
        self.outer_layout.addWidget(self.edit_config_file_btn, alignment=Qt.AlignRight)

    def _generate_config_layout(self, layout):
        cfg_text = QtWidgets.QLabel("Active config file:")

        self.cfg_line = QtWidgets.QLineEdit()
        # self.cfg_line.setMaximumWidth(1000)
        self.cfg_line.setMinimumHeight(30)
        self.cfg_line.setText(self.config)
        self.cfg_line.textChanged[str].connect(self.update_cfg)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.setMinimumHeight(30)
        browse_button.clicked.connect(self.browse_dir)

        layout.addWidget(cfg_text)
        layout.addWidget(self.cfg_line)
        layout.addWidget(browse_button)

    def _generate_layout_single_animal(self, layout):
        # Dynamic bodypart cropping
        self.crop_bodyparts = QtWidgets.QCheckBox("Dynamically crop bodyparts")
        self.crop_bodyparts.setCheckState(Qt.Unchecked)
        self.crop_bodyparts.stateChanged.connect(self.update_crop_choice)

        layout.addWidget(self.crop_bodyparts)

    def _generate_layout_visualization(self, layout):

        tmp_layout = QtWidgets.QVBoxLayout()

        # Plot Trajectories
        self.plot_trajectories = QtWidgets.QCheckBox("Plot trajectories")
        self.plot_trajectories.setCheckState(Qt.Unchecked)
        self.plot_trajectories.stateChanged.connect(self.update_plot_trajectory_choice)

        tmp_layout.addWidget(self.plot_trajectories)

        # Show trajectory plots
        self.show_trajectory_plots = QtWidgets.QCheckBox("Show trajectory plots")
        self.show_trajectory_plots.setCheckState(Qt.Unchecked)
        self.show_trajectory_plots.setEnabled(False)
        self.show_trajectory_plots.stateChanged.connect(self.update_showfigs_choice)

        tmp_layout.addWidget(self.show_trajectory_plots)

        layout.addLayout(tmp_layout)

        # Bodypart list
        self.bdpt_list_widget = QtWidgets.QListWidget()
        # self.bdpt_list_widget.setMaximumWidth(400)
        self.bdpt_list_widget.addItems(self.all_bodyparts)
        self.bdpt_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
        )
        self.bdpt_list_widget.setEnabled(False)
        self.bdpt_list_widget.itemSelectionChanged.connect(
            self.update_selected_bodyparts
        )
        tmp_layout = QtWidgets.QVBoxLayout()
        tmp_layout.addWidget(self.bdpt_list_widget)
        layout.addLayout(tmp_layout)

    def _generate_layout_data_processing(self, layout):
        # Save results as csv
        self.save_as_csv = QtWidgets.QCheckBox("Save result(s) as csv")
        self.save_as_csv.setCheckState(Qt.Unchecked)
        self.save_as_csv.stateChanged.connect(self.update_csv_choice)

        layout.addWidget(self.save_as_csv)

        # Filter predictions
        self.filter_predictions = QtWidgets.QCheckBox("Filter predictions")
        self.filter_predictions.setCheckState(Qt.Unchecked)
        self.filter_predictions.stateChanged.connect(self.update_filter_choice)

        layout.addWidget(self.filter_predictions)

    def _generate_layout_video_analysis(self, layout):

        self.videotype_widget = QComboBox()
        self.videotype_widget.setMaximumWidth(100)
        self.videotype_widget.setMinimumHeight(30)

        options = ["avi", "mp4", "mov"]
        self.videotype_widget.addItems(options)
        self.videotype_widget.setCurrentText(self.backend_variables["input_video_type"])
        self.videotype_widget.currentTextChanged.connect(self.update_videotype)

        layout.addWidget(self.videotype_widget)

        self.select_video_button = QtWidgets.QPushButton("Select videos")
        self.select_video_button.setMaximumWidth(200)
        self.select_video_button.setMinimumHeight(30)
        self.select_video_button.clicked.connect(self.select_video)

        layout.addWidget(self.select_video_button)

        self.selected_videos_text = QtWidgets.QLabel("")
        layout.addWidget(self.selected_videos_text)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(1)
        self.shuffle.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)
        self.trainingset.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.trainingset)

        # Overwrite analysis files
        self.overwrite_tracks = QtWidgets.QCheckBox("Overwrite tracks")
        self.overwrite_tracks.setCheckState(Qt.Unchecked)
        self.overwrite_tracks.stateChanged.connect(self.update_overwrite_tracks)

    def _generate_layout_multianimal_only_options(self, layout):

        tmp_layout = QtWidgets.QHBoxLayout()

        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_selector = QComboBox()
        self.tracker_type_selector.setMinimumHeight(30)
        self.tracker_type_selector.addItems(["skeleton", "box", "ellipse"])
        self.tracker_type_selector.setCurrentText(self.tracker_method)
        self.tracker_type_selector.currentTextChanged.connect(self.update_tracker_type)
        tmp_layout.addWidget(opt_text)
        tmp_layout.addWidget(self.tracker_type_selector)

        opt_text = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QSpinBox()
        self.num_animals_in_videos.setValue(len(self.cfg.get("individuals", 1)))
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.setMinimumHeight(30)
        tmp_layout.addWidget(opt_text)
        tmp_layout.addWidget(self.num_animals_in_videos)

        layout.addLayout(tmp_layout)


        tmp_layout = QtWidgets.QHBoxLayout()

        self.calibrate_assembly_checkbox = QtWidgets.QCheckBox("Calibrate assembly")
        self.calibrate_assembly_checkbox.setCheckState(Qt.Unchecked)
        self.calibrate_assembly_checkbox.stateChanged.connect(
            self.update_calibrate_assembly
        )
        tmp_layout.addWidget(self.calibrate_assembly_checkbox)


        self.assemble_with_ID_only_checkbox = QtWidgets.QCheckBox(
            "Assemble with ID only"
        )
        self.assemble_with_ID_only_checkbox.setCheckState(Qt.Unchecked)
        self.assemble_with_ID_only_checkbox.stateChanged.connect(
            self.update_assemble_with_ID_only
        )
        tmp_layout.addWidget(self.assemble_with_ID_only_checkbox)


        self.create_detections_video_checkbox = QtWidgets.QCheckBox(
            "Create video with all detections"
        )
        self.create_detections_video_checkbox.setCheckState(Qt.Unchecked)
        self.create_detections_video_checkbox.stateChanged.connect(
            self.update_create_video_detections
        )
        tmp_layout.addWidget(self.create_detections_video_checkbox)

        # Use ffprobe
        self.use_robustnframes = QtWidgets.QCheckBox("Robust frame reading")
        self.use_robustnframes.setCheckState(Qt.Unchecked)
        self.use_robustnframes.stateChanged.connect(self.update_robustnframes)
        tmp_layout.addWidget(self.use_robustnframes)

        layout.addLayout(tmp_layout)

    def update_robustnframes(self, state):
        if state == Qt.Checked:
            self.logger.info("Robust frame reading - use ffprobe ENABLED")
            self.backend_variables["robustnframes"] = True
        else:
            self.logger.info("Robust frame reading - use ffprobe DISABLED")
            self.backend_variables["robustnframes"] = False
            

    def update_create_video_detections(self, state):
        if state == Qt.Checked:
            self.backend_variables["create_video_all_detections"] = True
            self.logger.info("Create video with all detections ENABLED")
        else:
            self.backend_variables["create_video_all_detections"] = False
            self.logger.info("Create video with all detections DISABLED")

    def update_overwrite_tracks(self, state):
        if state == Qt.Checked:
            self.backend_variables["overwrite_tracks"] = True
            self.logger.info("Overwrite tracks ENABLED")
        else:
            self.backend_variables["overwrite_tracks"] = False
            self.logger.info("Overwrite tracks DISABLED")

    def update_assemble_with_ID_only(self, state):
        if state == Qt.Checked:
            self.backend_variables["assemble_with_ID_only"] = True
            self.logger.info("Assembly with ID only ENABLED")
        else:
            self.backend_variables["assemble_with_ID_only"] = False
            self.logger.info("Assembly with ID only DISABLED")

    def update_calibrate_assembly(self, state):
        if state == Qt.Checked:
            self.backend_variables["calibrate_assembly"] = True
            self.logger.info("Assembly calibration ENABLED")
        else:
            self.backend_variables["calibrate_assembly"] = False
            self.logger.info("Assembly calibration DISABLED")

    def update_videotype(self, vtype):
        self.logger.info(f"Looking for .{vtype} videos")
        self.backend_variables["input_video_type"] = vtype
        self.filelist.clear()
        self.selected_videos_text.setText("")
        self.select_video_button.setText("Select videos")

    def update_selected_bodyparts(self):
        selected_bodyparts = [
            item.text() for item in self.bdpt_list_widget.selectedItems()
        ]
        self.logger.info(
            f"Selected bodyparts for trajecories plotting:\n\t{selected_bodyparts}"
        )
        self.backend_variables["bodyparts_to_use"] = selected_bodyparts

    def update_tracker_type(self, method):
        self.logger.info(f"Using {method} tracker")
        self.tracker_method = method

    def update_csv_choice(self, s):
        if s == Qt.Checked:
            self.backend_variables["save_as_csv"] = True
            self.logger.info("Save results as CSV ENABLED")
        else:
            self.backend_variables["save_as_csv"] = False
            self.logger.info("Save results as CSV DISABLED")

    def update_filter_choice(self, s):
        if s == Qt.Checked:
            self.backend_variables["filter_data"] = True
            self.logger.info("Filtering predictions ENABLED")
        else:
            self.backend_variables["filter_data"] = False
            self.logger.info("Filtering predictions DISABLED")

    def update_showfigs_choice(self, s):
        if s == Qt.Checked:
            self.backend_variables["show_figures"] = True
            self.logger.info("Plots will show as pop ups.")
        else:
            self.backend_variables["show_figures"] = False
            self.logger.info("Plots will not show up.")

    def update_crop_choice(self, s):
        if s == Qt.Checked:
            self.backend_variables["dynamic_cropping"] = True
            self.logger.info("Dynamic bodypart cropping ENABLED.")
        else:
            self.backend_variables["dynamic_cropping"] = False
            self.logger.info("Dynamic bodypart cropping DISABLED.")

    def update_plot_trajectory_choice(self, s):
        if s == Qt.Checked:
            self.backend_variables["plot_trajectories"] = True
            self.bdpt_list_widget.setEnabled(True)
            self.show_trajectory_plots.setEnabled(True)
            self.logger.info("Plot trajectories ENABLED.")

        else:
            self.backend_variables["plot_trajectories"] = False
            self.bdpt_list_widget.setEnabled(False)
            self.show_trajectory_plots.setEnabled(False)
            self.show_trajectory_plots.setCheckState(Qt.Unchecked)
            self.logger.info("Plot trajectories DISABLED.")

    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config[0]:
            return
        self.config = config[0]
        self.cfg_line.setText(self.config)

    def select_video(self):
        cwd = self.config.split("/")[0:-1]
        cwd = "\\".join(cwd)
        filenames = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select video(s) to analyze",
            cwd,
            f"Video files (*.{self.videotype_widget.currentText()})",
        )

        if filenames:
            self.filelist.update(
                filenames[0]
            )  # Qt returns a tuple ( list of files, filetype )
            self.selected_videos_text.setText("%s videos selected" % len(self.filelist))
            self.select_video_button.setText("Add more videos")
            self.select_video_button.adjustSize()
            self.selected_videos_text.adjustSize()
            self.logger.info(f"Videos selected to analyze:\n{self.filelist}")

    def edit_config_file(self):

        if not self.config:
            return
        editor = ConfigEditor(self.config)
        editor.show()

    def analyze_videos(self):
        shuffle = self.shuffle.value()
        trainingsetindex = self.trainingset.value()

        save_as_csv = self.backend_variables["save_as_csv"]
        filter_data = self.backend_variables["filter_data"]
        videotype = self.backend_variables["input_video_type"]
        calibrate_assembly = self.backend_variables["calibrate_assembly"]
        assemble_with_ID_only = self.backend_variables["assemble_with_ID_only"]
        overwrite_tracks = self.backend_variables["overwrite_tracks"]
        create_video_all_detections = self.backend_variables[
            "create_video_all_detections"
        ]
        robustnframes = self.backend_variables["robustnframes"]
        track_method = self.tracker_method
        num_animals_in_videos = self.num_animals_in_videos.value()

        cropping = None
        dynamic_cropping_params = (False, 0.5, 10)

        if self.cfg["cropping"] == "True":
            cropping = self.cfg["x1"], self.cfg["x2"], self.cfg["y1"], self.cfg["y2"]

        if self.backend_variables["dynamic_cropping"]:
            dynamic_cropping_params = (True, 0.5, 10)

        # TODO
        # plug
        # tracker_method -> convert decections + stitch tracklets
        # bodyparts_to_use -> plot_trajectories only?
        # calibrate_assembly -> detections 2 tracklets
        # assemble_with_ID_only -> detections 2 tracklets

        scorername = deeplabcut.analyze_videos(
            self.config,
            self.filelist,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            gputouse=None,
            save_as_csv=save_as_csv,
            cropping=cropping,
            dynamic=dynamic_cropping_params,
            robust_nframes=robustnframes,
            auto_track=False,
            n_tracks=num_animals_in_videos,
            calibrate=calibrate_assembly,
            identity_only=assemble_with_ID_only,
        )

        if create_video_all_detections:
            deeplabcut.create_video_with_all_detections(
                config=self.config, 
                videos=self.filelist,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
            )

        if self.cfg["multianimalproject"]:
            deeplabcut.convert_detections2tracklets(
                self.config,
                self.filelist,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                overwrite=overwrite_tracks,
                calibrate=calibrate_assembly,
                identity_only=assemble_with_ID_only,
                track_method=track_method,
            )

            deeplabcut.stitch_tracklets(
                config_path=self.config,
                videos=self.filelist,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                n_tracks=num_animals_in_videos,
                track_method=track_method,
            )

        if filter_data:
            deeplabcut.filterpredictions(
                self.config,
                self.filelist,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtertype="median",
                windowlength=5,
                save_as_csv=save_as_csv,
            )

        if self.backend_variables["plot_trajectories"]:
            bdpts = self.bdpt_list_widget.selectedItems()
            self.logger.debug(f"Selected bodyparts for plot_trajectories: {bdpts}")
            showfig = self.backend_variables["show_figures"]
            deeplabcut.plot_trajectories(
                self.config,
                self.filelist,
                displayedbodyparts=bdpts,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtered=filter_data,
                showfigures=showfig,
            )
