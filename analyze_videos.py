import deeplabcut

from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QComboBox,
    QSpinBox,
)

from widgets import ConfigEditor
from components import (
    BodypartListWidget,
    DefaultTab,
    ShuffleSpinBox,
    TrainingSetSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
    _create_horizontal_layout,
    _create_vertical_layout,
)


class AnalyzeVideos(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(AnalyzeVideos, self).__init__(root, parent, h1_description)

        self.set_page()

    @property
    def files(self):
        self.video_selection_widget.files

    def set_page(self):

        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

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


        if self.root.is_multianimal:
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

        self.dynamic_cropping = False
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
        self.bodyparts_list_widget = BodypartListWidget(
            root=self.root, parent=self,
        )
        layout.addWidget(self.bodyparts_list_widget, Qt.AlignLeft)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingset = TrainingSetSpinBox(root=self.root, parent=self)

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
        self.tracker_type_widget = QComboBox()
        self.tracker_type_widget.addItems(["skeleton", "box", "ellipse"])
        self.tracker_type_widget.setCurrentText("skeleton")
        self.tracker_type_widget.currentTextChanged.connect(self.update_tracker_type)
        tmp_layout.addWidget(opt_text, 0, 0)
        tmp_layout.addWidget(self.tracker_type_widget, 0, 1)

        opt_text = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QSpinBox()
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        self.num_animals_in_videos.setMaximum(100)
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
            self.root.logger.info("Transformer tracking ENABLED")
        else:
            self.root.logger.info("Transformer tracking DISABLED")

    def update_robustnframes(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Robust frame reading - use ffprobe ENABLED")
        else:
            self.root.logger.info("Robust frame reading - use ffprobe DISABLED")

    def update_create_video_detections(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Create video with all detections ENABLED")
        else:
            self.root.logger.info("Create video with all detections DISABLED")

    def update_overwrite_tracks(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Overwrite tracks ENABLED")
        else:
            self.root.logger.info("Overwrite tracks DISABLED")

    def update_assemble_with_ID_only(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Assembly with ID only ENABLED")
        else:
            self.root.logger.info("Assembly with ID only DISABLED")

    def update_calibrate_assembly(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Assembly calibration ENABLED")
        else:
            self.root.logger.info("Assembly calibration DISABLED")

    def update_tracker_type(self, method):
        self.root.logger.info(f"Using {method.upper()} tracker")

    def update_csv_choice(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Save results as CSV ENABLED")
        else:
            self.root.logger.info("Save results as CSV DISABLED")

    def update_filter_choice(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Filtering predictions ENABLED")
        else:
            self.root.logger.info("Filtering predictions DISABLED")

    def update_showfigs_choice(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Plots will show as pop ups.")
        else:
            self.root.logger.info("Plots will not show up.")

    def update_crop_choice(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Dynamic bodypart cropping ENABLED.")
            self.dynamic_cropping = True
        else:
            self.root.logger.info("Dynamic bodypart cropping DISABLED.")
            self.dynamic_cropping = False

    def update_plot_trajectory_choice(self, s):
        if s == Qt.Checked:
            self.bodyparts_list_widget.show()
            self.bodyparts_list_widget.setEnabled(True)
            self.show_trajectory_plots.setEnabled(True)
            self.root.logger.info("Plot trajectories ENABLED.")

        else:
            self.bodyparts_list_widget.hide()
            self.bodyparts_list_widget.setEnabled(False)
            self.show_trajectory_plots.setEnabled(False)
            self.show_trajectory_plots.setCheckState(Qt.Unchecked)
            self.root.logger.info("Plot trajectories DISABLED.")

    def edit_config_file(self):

        if not self.root.config:
            return
        editor = ConfigEditor(self.root.config)
        editor.show()

    def analyze_videos(self):

        config = self.root.config
        shuffle = self.root.shuffle_value
        trainingsetindex = self.trainingset.value()

        videos = list(self.files)
        save_as_csv = self.save_as_csv.checkState() == Qt.Checked
        filter_data = self.filter_predictions.checkState() == Qt.Checked
        videotype = self.video_selection_widget.videotype_widget.currentText()
        calibrate_assembly = self.calibrate_assembly_checkbox.checkState() == Qt.Checked
        assemble_with_ID_only = self.assemble_with_ID_only_checkbox.checkState() == Qt.Checked
        overwrite_tracks = self.overwrite_tracks.checkState() == Qt.Checked
        create_video_all_detections = self.create_detections_video_checkbox.checkState() == Qt.Checked
        robustnframes = self.use_robustnframes.checkState() == Qt.Checked
        use_transformer_tracking = self.use_transformer_tracking_checkbox.checkState() == Qt.Checked
        track_method = self.tracker_type_widget.currentText() if self.root.is_multianimal else ""
        num_animals_in_videos = self.num_animals_in_videos.value()

        cropping = None
        dynamic_cropping_params = (False, 0.5, 10)

        if self.root.cfg["cropping"] == "True":
            cropping = self.root.cfg["x1"], self.root.cfg["x2"], self.root.cfg["y1"], self.root.cfg["y2"]

        if self.dynamic_cropping:
            dynamic_cropping_params = (True, 0.5, 10)

        scorername = deeplabcut.analyze_videos(
            config,
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
                config,
                videos=videos,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
            )

        if self.root.is_multianimal:
            deeplabcut.convert_detections2tracklets(
                config,
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
                    config,
                    videos=videos,
                    videotype=videotype,
                    shuffle=shuffle,
                    trainingsetindex=trainingsetindex,
                    n_tracks=num_animals_in_videos,
                    track_method=track_method,
                )

        if filter_data:
            deeplabcut.filterpredictions(
                config,
                videos=videos,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtertype="median",
                windowlength=5,
                save_as_csv=save_as_csv,
            )

        if self.plot_trajectories.checkState() == Qt.Checked:
            bdpts = self.bodyparts_list_widget.selected_bodyparts
            self.logger.debug(f"Selected bodyparts for plot_trajectories: {bdpts}")
            showfig = self.show_trajectory_plots.checkState() == Qt.Checked
            deeplabcut.plot_trajectories(
                config,
                videos=videos,
                displayedbodyparts=bdpts,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtered=filter_data,
                showfigures=showfig,
            )
