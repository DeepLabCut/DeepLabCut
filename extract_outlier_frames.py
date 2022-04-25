from PySide2.QtWidgets import QWidget, QComboBox, QSpinBox
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

import deeplabcut
from deeplabcut import utils
from deeplabcut.utils import auxiliaryfunctions

from components import DefaultTab, VideoSelectionWidget, _create_horizontal_layout, _create_label_widget



class ExtractOutlierFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(ExtractOutlierFrames, self).__init__(root, parent, h1_description)
        self.filelist = []

        self.set_page()

    @property
    def files(self):
        self.video_selection_widget.files

    def set_page(self):

        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.main_layout.addWidget(
            _create_label_widget("Attributes", "font:bold")
        )
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)

        self._generate_multianimal_options(self.layout_attributes)

        self.main_layout.addLayout(self.layout_attributes)


        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setSpacing(20)
        self.layout_attributes.setContentsMargins(0, 0, 40, 0)

        self.main_layout.addWidget(
            _create_label_widget("Frame extraction options", "font:bold")
        )
        self.layout_extraction_options = _create_horizontal_layout()
        self._generate_layout_extraction_options(self.layout_extraction_options)
        self.main_layout.addLayout(self.layout_extraction_options)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.setContentsMargins(0, 40, 40, 40)
        self.ok_button.clicked.connect(self.extract_outlier_frames)

        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

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

    def _generate_multianimal_options(self, layout):
        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_widget = QComboBox()
        self.tracker_type_widget.setMinimumWidth(150)
        self.tracker_type_widget.addItems(["skeleton", "box", "ellipse"])
        self.tracker_type_widget.setCurrentText("skeleton")
        self.tracker_type_widget.currentTextChanged.connect(self.update_tracker_type)

        layout.addWidget(opt_text)
        layout.addWidget(self.tracker_type_widget)
        if not self.root.is_multianimal:
            opt_text.hide()
            self.tracker_type_widget.hide()

    def _generate_layout_extraction_options(self, layout):

        opt_text = QtWidgets.QLabel("Specify the algorithm")
        self.outlier_algorithm_widget = QComboBox()
        self.outlier_algorithm_widget.setMinimumWidth(150)
        options = ["jump", "fitting", "uncertain", "manual"]
        self.outlier_algorithm_widget.addItems(options)
        self.outlier_algorithm_widget.setCurrentText("jump")
        self.outlier_algorithm_widget.currentTextChanged.connect(self.update_outlier_algorithm)

        layout.addWidget(opt_text)
        layout.addWidget(self.outlier_algorithm_widget)

    def update_tracker_type(self, method):
        self.root.logger.info(f"Using {method.upper()} tracker")

    def update_outlier_algorithm(self, algorithm):
        self.root.logger.info(f"Using {algorithm.upper()} algorithm for frame extraction")

    def extract_outlier_frames(self):
        config = self.root.config
        shuffle = self.root.shuffle_value
        trainingsetindex = self.trainingset.value()
        videos = self.files
        videotype = self.video_selection_widget.videotype_widget.currentText()
        outlieralgorithm = self.outlier_algorithm_widget.currentText()
        track_method = ""

        if self.root.is_multianimal:
            track_method = self.tracker_type_widget.currentText()

        self.root.logger.debug(
        f"""Running extract outlier frames with options:
        config: {config},
        shuffle: {shuffle},
        trainingset index: {trainingsetindex},
        videos: {videos},
        videotype: {videotype},
        outlier algorithm: {outlieralgorithm},
        track method: {track_method}
        """)
        deeplabcut.extract_outlier_frames(
            config=config,
            videos=videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            outlieralgorithm=outlieralgorithm,
            track_method=track_method,
        )
