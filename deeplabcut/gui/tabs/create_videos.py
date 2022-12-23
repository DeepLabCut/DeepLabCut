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
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from deeplabcut.gui.components import (
    BodypartListWidget,
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)

import deeplabcut


class CreateVideos(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(CreateVideos, self).__init__(root, parent, h1_description)

        self.bodyparts_to_use = self.root.all_bodyparts
        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):

        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        tmp_layout = _create_horizontal_layout()

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout(margins=(0, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        tmp_layout.addLayout(self.layout_attributes)

        self.layout_multianimal = _create_horizontal_layout()

        if self.root.is_multianimal:
            self._generate_layout_multianimal(self.layout_multianimal)
            tmp_layout.addLayout(self.layout_multianimal)

        self.main_layout.addLayout(tmp_layout)

        self.main_layout.addWidget(
            _create_label_widget("Video Parameters", "font:bold")
        )
        self.layout_video_parameters = _create_vertical_layout()
        self._generate_layout_video_parameters(self.layout_video_parameters)
        self.main_layout.addLayout(self.layout_video_parameters)

        self.sk_button = QtWidgets.QPushButton("Build skeleton")
        self.sk_button.clicked.connect(self.build_skeleton)
        self.main_layout.addWidget(self.sk_button, alignment=Qt.AlignRight)

        self.run_button = QtWidgets.QPushButton("Create videos")
        self.run_button.clicked.connect(self.create_videos)
        self.main_layout.addWidget(self.run_button, alignment=Qt.AlignRight)

    def _generate_layout_multianimal(self, layout):
        tmp_text = QtWidgets.QLabel("Color keypoints by:")
        self.color_by_widget = QtWidgets.QComboBox()
        self.color_by_widget.addItems(["bodypart", "individual"])
        self.color_by_widget.setCurrentText("bodypart")
        self.color_by_widget.currentTextChanged.connect(self.update_color_by)

        layout.addWidget(tmp_text)
        layout.addWidget(self.color_by_widget)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

        # Overwrite videos
        self.overwrite_videos = QtWidgets.QCheckBox("Overwrite videos")
        self.overwrite_videos.setCheckState(Qt.Unchecked)
        self.overwrite_videos.stateChanged.connect(self.update_overwrite_videos)

        layout.addWidget(self.overwrite_videos)

    def _generate_layout_video_parameters(self, layout):

        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        # Trail Points
        opt_text = QtWidgets.QLabel("Specify the number of trail points")
        self.trail_points = QtWidgets.QSpinBox()
        self.trail_points.setValue(0)
        tmp_layout.addWidget(opt_text)
        tmp_layout.addWidget(self.trail_points)

        layout.addLayout(tmp_layout)

        tmp_layout = _create_vertical_layout(margins=(0, 0, 0, 0))

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

        nested_tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))
        nested_tmp_layout.addLayout(tmp_layout)

        tmp_layout = _create_vertical_layout(margins=(0, 0, 0, 0))

        # Bodypart list
        self.bodyparts_list_widget = BodypartListWidget(
            root=self.root,
            parent=self,
        )
        nested_tmp_layout.addWidget(self.bodyparts_list_widget, Qt.AlignLeft)

        tmp_layout.addLayout(nested_tmp_layout, Qt.AlignLeft)

        layout.addLayout(tmp_layout, Qt.AlignLeft)

    def update_high_quality_video(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"High quality {s}.")

    def update_plot_trajectory_choice(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Plot trajectories {s}.")

    def update_selected_bodyparts(self):
        selected_bodyparts = [
            item.text() for item in self.bodyparts_list_widget.selectedItems()
        ]
        self.root.logger.info(
            f"Selected bodyparts for plotting:\n\t{selected_bodyparts}"
        )
        self.bodyparts_to_use = selected_bodyparts

    def update_use_all_bodyparts(self, s):
        if s == Qt.Checked:
            self.bodyparts_list_widget.setEnabled(False)
            self.bodyparts_list_widget.hide()
            self.root.logger.info("Plot all bodyparts ENABLED.")

        else:
            self.bodyparts_list_widget.setEnabled(True)
            self.bodyparts_list_widget.show()
            self.root.logger.info("Plot all bodyparts DISABLED.")

    def update_use_filtered_data(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Use filtered data {s}")

    def update_draw_skeleton(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Draw skeleton {s}")

    def update_overwrite_videos(self, state):
        s = "ENABLED" if state == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Overwrite videos {s}")

    def update_color_by(self, text):
        self.root.logger.info(f"Coloring keypoints in videos by {text}")

    def update_filter_choice(self, rb):
        self.filtered = rb.text() == "Yes"

    def update_video_slow_choice(self, rb):
        self.slow = rb.text() == "Yes"

    def update_draw_skeleton_choice(self, rb):
        self.draw = rb.text() == "Yes"

    def create_videos(self):
        config = self.root.config
        shuffle = self.root.shuffle_value
        videos = self.files
        trailpoints = self.trail_points.value()
        if hasattr(self, "color_by_widget"):
            # Multianimal scenario.
            # Color is based on individual or bodypart.
            color_by = self.color_by_widget.currentText()
        else:
            # Single animal scenario.
            # Color is based on bodypart.
            color_by = "bodypart"
        filtered = bool(self.use_filtered_data_checkbox.checkState())

        bodyparts = "all"
        if (
            len(self.bodyparts_to_use) != 0
            and self.plot_all_bodyparts.checkState() == Qt.Checked
        ):
            bodyparts = self.bodyparts_to_use

        deeplabcut.create_labeled_video(
            config=config,
            videos=videos,
            shuffle=shuffle,
            filtered=filtered,
            save_frames=bool(self.create_high_quality_video.checkState()),
            displayedbodyparts=bodyparts,
            draw_skeleton=bool(self.draw_skeleton_checkbox.checkState()),
            trailpoints=trailpoints,
            color_by=color_by,
        )
        self.root.writer.write("Labeled videos created.")

        if self.plot_trajectories.checkState():
            deeplabcut.plot_trajectories(
                config=config,
                videos=videos,
                shuffle=shuffle,
                filtered=filtered,
                displayedbodyparts=bodyparts,
            )

    def build_skeleton(self, *args):
        from deeplabcut.gui.widgets import SkeletonBuilder

        SkeletonBuilder(self.root.config)
