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
from functools import partial
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.utils import move_to_separate_thread

import deeplabcut


class UnsupervizedIdTracking(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(UnsupervizedIdTracking, self).__init__(root, parent, h1_description)

        self._set_page()

    @property
    def files(self):
        return self.root.video_files

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.run_transformer_button = QtWidgets.QPushButton("Run transformer")
        self.run_transformer_button.clicked.connect(self.run_transformer)

        self.main_layout.addWidget(self.run_transformer_button, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(deeplabcut.transformer_reID.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self, layout):
        # Shuffle
        shuffle_label = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Tracker Type
        trackingtype_label = QtWidgets.QLabel("Tracking method")
        self.tracker_type_widget = QtWidgets.QComboBox()
        self.tracker_type_widget.addItems(["ellipse", "box"])
        self.tracker_type_widget.currentTextChanged.connect(self.log_tracker_type)

        # Num animals
        num_animals_label = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QtWidgets.QSpinBox()
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.valueChanged.connect(self.log_num_animals)

        # Num triplets
        num_triplets_label = QtWidgets.QLabel("Number of triplets")
        self.num_triplets = QtWidgets.QSpinBox()
        self.num_triplets.setMaximum(1000000)
        self.num_triplets.setValue(1000)
        self.num_triplets.valueChanged.connect(self.log_num_triplets)

        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(trackingtype_label, 0, 2)
        layout.addWidget(self.tracker_type_widget, 0, 3)
        layout.addWidget(num_animals_label, 1, 0)
        layout.addWidget(self.num_animals_in_videos, 1, 1)
        layout.addWidget(num_triplets_label, 1, 2)
        layout.addWidget(self.num_triplets, 1, 3)
        # layout.addWidget()

    def log_tracker_type(self, tracker):
        self.root.logger.info(f"Tracker type set to {tracker.upper()}")

    def log_num_animals(self, value):
        self.root.logger.info(f"Num animals set to {value}")

    def log_num_triplets(self, value):
        self.root.logger.info(f"Num triplets set to {value}")

    def run_transformer(self):
        config = self.root.config
        videos = [v for v in self.files]
        videotype = self.video_selection_widget.videotype_widget.currentText()
        n_tracks = self.num_animals_in_videos.value()
        shuffle = self.shuffle.value()
        track_method = self.tracker_type_widget.currentText()

        func = partial(
            deeplabcut.transformer_reID,
            config=config,
            videos=videos,
            videotype=videotype,
            n_tracks=n_tracks,
            shuffle=shuffle,
            track_method=track_method,
        )
        self.worker, self.thread = move_to_separate_thread(func)
        self.worker.finished.connect(
            lambda: self.run_transformer_button.setEnabled(True)
        )
        self.worker.finished.connect(lambda: self.root._progress_bar.hide())
        self.thread.start()
        self.run_transformer_button.setEnabled(False)
        self.root._progress_bar.show()
