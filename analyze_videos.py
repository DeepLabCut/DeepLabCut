import logging

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QComboBox,
    QSpinBox,
)

from widgets import ConfigEditor
from components import (
    DefaultTab,
    _create_grid_layout,
    _create_label_widget,
    _create_horizontal_layout,
    _create_vertical_layout,
)


class AnalyzeVideos(DefaultTab):
    def __init__(self, parent, tab_heading):
        super(AnalyzeVideos, self).__init__(parent, tab_heading)

        self.filelist = set()
        
        if self.is_multianimal:
            self.tracker_method = self.cfg.get("default_track_method", "ellipse")

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
            "robustnframes": False,  # Use ffprobe
            "use_transformer_tracking": False,
        }

        self.set_page()

    def set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.layout_video_selection = _create_horizontal_layout()
        self._generate_layout_video_analysis(self.layout_video_selection)
        self.main_layout.addLayout(self.layout_video_selection)

        tmp_layout = _create_horizontal_layout()

        self.main_layout.addWidget(
            _create_label_widget("Attributes", "font:bold")
        )
        self.layout_attributes = _create_grid_layout()
        self._generate_layout_attributes(self.layout_attributes)
        tmp_layout.addLayout(self.layout_attributes)

        # Single / Multi animal Only Layouts
        self.layout_single_animal = _create_horizontal_layout()

        self.layout_multi_animal = _create_horizontal_layout()


        if self.cfg["multianimalproject"]:
            self._generate_layout_multianimal_only_options(self.layout_multi_animal)
            tmp_layout.addLayout(self.layout_multi_animal)
        else:
            self._generate_layout_single_animal(self.layout_single_animal)
            tmp_layout.addLayout(self.layout_single_animal)

        self.main_layout.addLayout(tmp_layout)

        self.main_layout.addWidget(_create_label_widget("", "font:bold"))
        self.layout_other_options = _create_vertical_layout()
        self._generate_layout_other_options(self.layout_other_options)
        self.main_layout.addLayout(self.layout_other_options)

        self.analyze_videos_btn = QtWidgets.QPushButton("Analyze Videos")
        self.analyze_videos_btn.clicked.connect(self.analyze_videos)

        self.edit_config_file_btn = QtWidgets.QPushButton("Edit config.yaml")
        self.edit_config_file_btn.clicked.connect(self.edit_config_file)

        self.main_layout.addWidget(self.analyze_videos_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.edit_config_file_btn, alignment=Qt.AlignRight)

    def _generate_layout_single_animal(self, layout):
        # Dynamic bodypart cropping
        self.crop_bodyparts = QtWidgets.QCheckBox("Dynamically crop bodyparts")
        self.crop_bodyparts.setCheckState(Qt.Unchecked)
        self.crop_bodyparts.stateChanged.connect(self.update_crop_choice)

        layout.addWidget(self.crop_bodyparts)

    def _generate_layout_other_options(self, layout):

        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        # Save results as csv
        self.save_as_csv = QtWidgets.QCheckBox("Save result(s) as csv")
        self.save_as_csv.setCheckState(Qt.Unchecked)
        self.save_as_csv.stateChanged.connect(self.update_csv_choice)

        tmp_layout.addWidget(self.save_as_csv)

        # Filter predictions
        self.filter_predictions = QtWidgets.QCheckBox("Filter predictions")
        self.filter_predictions.setCheckState(Qt.Unchecked)
        self.filter_predictions.stateChanged.connect(self.update_filter_choice)

        tmp_layout.addWidget(self.filter_predictions)


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
        self.bdpt_list_widget.setMaximumWidth(500)
        self.bdpt_list_widget.addItems(self.all_bodyparts)
        self.bdpt_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
        )
        # self.bdpt_list_widget.selectAll()
        self.bdpt_list_widget.setEnabled(False)
        self.bdpt_list_widget.itemSelectionChanged.connect(
            self.update_selected_bodyparts
        )
        layout.addWidget(self.bdpt_list_widget, Qt.AlignLeft)

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
        self.select_video_button.clicked.connect(self.select_videos)

        layout.addWidget(self.select_video_button)

        self.selected_videos_text = QtWidgets.QLabel("")
        layout.addWidget(self.selected_videos_text)

        self.clear_videos = QtWidgets.QPushButton("Clear selection")
        self.clear_videos.clicked.connect(self.clear_selected_videos)
        layout.addWidget(self.clear_videos, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(1)
        self.shuffle.setMinimumHeight(30)

        layout.addWidget(opt_text, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)
        self.trainingset.setMinimumHeight(30)

        layout.addWidget(opt_text, 1, 0)
        layout.addWidget(self.trainingset, 1, 1)

        # Overwrite analysis files
        self.overwrite_tracks = QtWidgets.QCheckBox("Overwrite tracks")
        self.overwrite_tracks.setCheckState(Qt.Unchecked)
        self.overwrite_tracks.stateChanged.connect(self.update_overwrite_tracks)

        layout.addWidget(self.overwrite_tracks, 0, 2)

    def _generate_layout_multianimal_only_options(self, layout):
        
        tmp_layout = QtWidgets.QGridLayout()

        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_selector = QComboBox()
        self.tracker_type_selector.setMinimumHeight(30)
        self.tracker_type_selector.addItems(["skeleton", "box", "ellipse"])
        self.tracker_type_selector.setCurrentText(self.tracker_method)
        self.tracker_type_selector.currentTextChanged.connect(self.update_tracker_type)
        tmp_layout.addWidget(opt_text, 0, 0)
        tmp_layout.addWidget(self.tracker_type_selector, 0, 1)

        opt_text = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QSpinBox()
        self.num_animals_in_videos.setValue(len(self.cfg.get("individuals", 1)))
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.setMinimumHeight(30)
        tmp_layout.addWidget(opt_text, 1, 0)
        tmp_layout.addWidget(self.num_animals_in_videos, 1, 1)

        # layout.addLayout(tmp_layout)

        # tmp_layout = QtWidgets.QGridLayout()

        self.calibrate_assembly_checkbox = QtWidgets.QCheckBox("Calibrate assembly")
        self.calibrate_assembly_checkbox.setCheckState(Qt.Unchecked)
        self.calibrate_assembly_checkbox.stateChanged.connect(
            self.update_calibrate_assembly
        )
        tmp_layout.addWidget(self.calibrate_assembly_checkbox, 0, 2)

        self.assemble_with_ID_only_checkbox = QtWidgets.QCheckBox(
            "Assemble with ID only"
        )
        self.assemble_with_ID_only_checkbox.setCheckState(Qt.Unchecked)
        self.assemble_with_ID_only_checkbox.stateChanged.connect(
            self.update_assemble_with_ID_only
        )
        tmp_layout.addWidget(self.assemble_with_ID_only_checkbox, 1, 2)

        self.use_transformer_tracking_checkbox = QtWidgets.QCheckBox(
            "Use tranformer tracking"
        )
        self.use_transformer_tracking_checkbox.setCheckState(Qt.Unchecked)
        self.use_transformer_tracking_checkbox.stateChanged.connect(
            self.update_use_transformer_tracking
        )
        tmp_layout.addWidget(self.use_transformer_tracking_checkbox, 0, 3)

        self.create_detections_video_checkbox = QtWidgets.QCheckBox(
            "Create video with all detections"
        )
        self.create_detections_video_checkbox.setCheckState(Qt.Unchecked)
        self.create_detections_video_checkbox.stateChanged.connect(
            self.update_create_video_detections
        )
        tmp_layout.addWidget(self.create_detections_video_checkbox, 1, 3)

        # Use ffprobe
        self.use_robustnframes = QtWidgets.QCheckBox("Robust frame reading")
        self.use_robustnframes.setCheckState(Qt.Unchecked)
        self.use_robustnframes.stateChanged.connect(self.update_robustnframes)
        tmp_layout.addWidget(self.use_robustnframes, 0, 4)

        layout.addLayout(tmp_layout)

    def update_use_transformer_tracking(self, state):
        if state == Qt.Checked:
            self.logger.info("Transformer tracking ENABLED")
            self.backend_variables["use_transformer_tracking"] = True
        else:
            self.logger.info("Transformer tracking DISABLED")
            self.backend_variables["use_transformer_tracking"] = False

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

    def select_videos(self):
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

    def clear_selected_videos(self):
        self.selected_videos_text.setText("")
        self.select_video_button.setText("Select videos")
        self.filelist.clear()
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

        videos = list(self.filelist)
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
        use_transformer_tracking = self.backend_variables["use_transformer_tracking"]
        track_method = self.tracker_method
        num_animals_in_videos = self.num_animals_in_videos.value()

        cropping = None
        dynamic_cropping_params = (False, 0.5, 10)

        if self.cfg["cropping"] == "True":
            cropping = self.cfg["x1"], self.cfg["x2"], self.cfg["y1"], self.cfg["y2"]

        if self.backend_variables["dynamic_cropping"]:
            dynamic_cropping_params = (True, 0.5, 10)

        scorername = deeplabcut.analyze_videos(
            self.config,
            videos=videos,
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
                videos=videos,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
            )

        if self.cfg["multianimalproject"]:
            deeplabcut.convert_detections2tracklets(
                self.config,
                videos=videos,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                overwrite=overwrite_tracks,
                calibrate=calibrate_assembly,
                identity_only=assemble_with_ID_only,
                track_method=track_method,
            )

            if use_transformer_tracking:
                raise NotImplementedError(
                    "Transformer has not been integrated to GUI yet"
                )
                # TODO: Plug in code, when codebase stable.
            else:
                deeplabcut.stitch_tracklets(
                    config_path=self.config,
                    videos=videos,
                    videotype=videotype,
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    n_tracks=num_animals_in_videos,
                    track_method=track_method,
                )

        if filter_data:
            deeplabcut.filterpredictions(
                self.config,
                videos=videos,
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
                videos=videos,
                displayedbodyparts=bdpts,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtered=filter_data,
                showfigures=showfig,
            )
