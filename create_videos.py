import deeplabcut

from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QComboBox,
    QSpinBox,
)

from components import (
    DefaultTab,
    VideoSelectionWidget,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)


class CreateVideos(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(CreateVideos, self).__init__(root, parent, h1_description)

        self.filelist = set()
        self.bodyparts_to_use = self.root.all_bodyparts

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
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)
        tmp_layout.addLayout(self.layout_attributes)

        self.layout_multi_animal = _create_horizontal_layout()

        if self.root.is_multianimal:
            self._generate_layout_multianimal_only_options(self.layout_multi_animal)
            tmp_layout.addLayout(self.layout_multi_animal)

        self.main_layout.addLayout(tmp_layout)

        self.main_layout.addWidget(
            _create_label_widget("Video Parameters", "font:bold")
        )
        self.layout_video_parameters = _create_vertical_layout()
        self._generate_layout_video_parameters(self.layout_video_parameters)
        self.main_layout.addLayout(self.layout_video_parameters)

        self.run_button = QtWidgets.QPushButton("Create videos")
        self.run_button.clicked.connect(self.create_videos)
        self.main_layout.addWidget(self.run_button, alignment=Qt.AlignRight)

    def _generate_layout_multianimal_only_options(self, layout):
        tmp_text = QtWidgets.QLabel("Color keypoints by:")
        self.color_by_widget = QComboBox()
        self.color_by_widget.setMaximumWidth(150)
        self.color_by_widget.addItems(["bodypart", "individual"])
        self.color_by_widget.setCurrentText("bodypart")
        self.color_by_widget.currentTextChanged.connect(self.update_color_by)

        layout.addWidget(tmp_text)
        layout.addWidget(self.color_by_widget)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(self.root.shuffle_value)
        self.shuffle.valueChanged.connect(self.root.update_shuffle)


        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)

        layout.addWidget(opt_text)
        layout.addWidget(self.trainingset)

        # Overwrite videos
        self.overwrite_videos = QtWidgets.QCheckBox("Overwrite videos")
        self.overwrite_videos.setCheckState(Qt.Unchecked)
        self.overwrite_videos.stateChanged.connect(self.update_overwrite_videos)

        layout.addWidget(self.overwrite_videos)

    def _generate_layout_video_parameters(self, layout):

        tmp_layout = _create_horizontal_layout()

        # Trail Points
        opt_text = QtWidgets.QLabel("Specify the number of trail points")
        self.trail_points = QSpinBox()
        self.trail_points.setValue(0)
        self.trail_points.setMinimumWidth(100)
        tmp_layout.addWidget(opt_text)
        tmp_layout.addWidget(self.trail_points)

        layout.addLayout(tmp_layout)

        tmp_layout = _create_vertical_layout()

        # Plot all bodyparts
        self.plot_all_bodyparts = QtWidgets.QCheckBox("Plot all bodyparts")
        self.plot_all_bodyparts.setCheckState(Qt.Checked)
        self.plot_all_bodyparts.stateChanged.connect(self.update_use_all_bodyparts)
        tmp_layout.addWidget(self.plot_all_bodyparts)

        # Skeleton
        self.draw_skeleton_checkbox = QtWidgets.QCheckBox("Draw skeleton")
        self.draw_skeleton_checkbox.setCheckState(Qt.Checked)
        self.draw_skeleton_checkbox.stateChanged.connect(self.update_draw_skeleton)
        tmp_layout.addWidget(self.draw_skeleton_checkbox)

        # Filtered data
        self.use_filtered_data_checkbox = QtWidgets.QCheckBox("Use filtered data")
        self.use_filtered_data_checkbox.setCheckState(Qt.Unchecked)
        self.use_filtered_data_checkbox.stateChanged.connect(
            self.update_use_filtered_data
        )
        tmp_layout.addWidget(self.use_filtered_data_checkbox)

        # Plot trajectories
        self.plot_trajectories = QtWidgets.QCheckBox("Plot trajectories")
        self.plot_trajectories.setCheckState(Qt.Unchecked)
        self.plot_trajectories.stateChanged.connect(self.update_plot_trajectory_choice)
        tmp_layout.addWidget(self.plot_trajectories)

        # High quality video
        self.create_high_quality_video = QtWidgets.QCheckBox(
            "High quality video (slow)"
        )
        self.create_high_quality_video.setCheckState(Qt.Unchecked)
        self.create_high_quality_video.stateChanged.connect(
            self.update_high_quality_video
        )
        tmp_layout.addWidget(self.create_high_quality_video)

        nested_tmp_layout = _create_horizontal_layout()
        nested_tmp_layout.addLayout(tmp_layout)

        tmp_layout = _create_vertical_layout()
        # Bodypart list
        self.bdpt_list_widget = QtWidgets.QListWidget()
        # self.bdpt_list_widget.setMaximumWidth(500)
        self.bdpt_list_widget.addItems(self.root.all_bodyparts)
        self.bdpt_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
        )
        # self.bdpt_list_widget.selectAll()
        self.bdpt_list_widget.setEnabled(False)
        self.bdpt_list_widget.itemSelectionChanged.connect(
            self.update_selected_bodyparts
        )
        nested_tmp_layout.addWidget(self.bdpt_list_widget, Qt.AlignLeft)

        tmp_layout.addLayout(nested_tmp_layout, Qt.AlignLeft)

        layout.addLayout(tmp_layout, Qt.AlignLeft)

    def update_high_quality_video(self, s):
        if s == Qt.Checked:
            self.root.logger.info("High quality ENABLED.")

        else:
            self.root.logger.info("High quality DISABLED.")

    def update_plot_trajectory_choice(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Plot trajectories ENABLED.")

        else:
            self.root.logger.info("Plot trajectories DISABLED.")

    def update_selected_bodyparts(self):
        selected_bodyparts = [
            item.text() for item in self.bdpt_list_widget.selectedItems()
        ]
        self.root.logger.info(f"Selected bodyparts for plotting:\n\t{selected_bodyparts}")
        self.bodyparts_to_use = selected_bodyparts

    def update_use_all_bodyparts(self, s):
        if s == Qt.Checked:
            self.bdpt_list_widget.setEnabled(False)
            # self.bdpt_list_widget.selectAll()
            self.root.logger.info("Plot all bodyparts ENABLED.")

        else:
            self.bdpt_list_widget.setEnabled(True)
            self.root.logger.info("Plot all bodyparts DISABLED.")

    def update_use_filtered_data(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Use filtered data ENABLED")
        else:
            self.root.logger.info("Use filtered data DISABLED")

    def update_draw_skeleton(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Draw skeleton ENABLED")
        else:
            self.root.logger.info("Draw skeleton DISABLED")

    def update_overwrite_videos(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Overwrite videos ENABLED")
        else:
            self.root.logger.info("Overwrite videos DISABLED")

    def update_color_by(self, text):
        self.root.logger.info(f"Coloring keypoints in videos by {text}")

    def update_filter_choice(self, rb):
        if rb.text() == "Yes":
            self.filtered = True
        else:
            self.filtered = False

    def update_video_slow_choice(self, rb):
        if rb.text() == "Yes":
            self.slow = True
        else:
            self.slow = False

    def update_draw_skeleton_choice(self, rb):
        if rb.text() == "Yes":
            self.draw = True
        else:
            self.draw = False

    def create_videos(self):

        config = self.root.config
        shuffle = self.root.shuffle_value
        trainingsetindex = self.trainingset.value()
        videos = self.files
        bodyparts = "all"
        videotype = self.videotype_widget.currentText()
        trailpoints = self.trail_points.value()
        color_by = self.color_by_widget.currentText()

        filtered = True
        if self.use_filtered_data_checkbox.checkState() == False:
            filtered = False

        draw_skeleton = True
        if self.draw_skeleton_checkboxx.checkState() == False:
            draw_skeleton = False

        slow_video = True
        if self.create_high_quality_video.checkState() == False:
            slow_video = False

        plot_trajectories = True
        if self.plot_trajectories.checkState() == False:
            plot_trajectories = False

        bodyparts = "all"
        if  len(self.bodyparts_to_use)!=0 and self.plot_all_bodyparts.checkState()==Qt.Checked:
            bodyparts = self.bodyparts_to_use

        deeplabcut.create_labeled_video(
            config=config,
            videos=videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            filtered=filtered,
            save_frames=slow_video,
            displayedbodyparts=bodyparts,
            draw_skeleton=draw_skeleton,
            trailpoints=trailpoints,
            color_by=color_by,
        )

        if plot_trajectories:
            deeplabcut.plot_trajectories(
                config=config,
                videos=videos,
                videotype=videotype,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                filtered=filtered,
                displayedbodyparts=bodyparts,
            )
