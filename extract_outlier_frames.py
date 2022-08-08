from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from dlc_params import DLC_Params
from components import (
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_horizontal_layout,
    _create_label_widget,
)

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions



def refine_labels(config, multianimal=False):
    """
    Refines the labels of the outlier frames extracted from the analyzed videos.\n Helps in augmenting the training dataset.
    Use the function ``analyze_video`` to analyze a video and extracts the outlier frames using the function
    ``extract_outlier_frames`` before refining the labels.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Screens : int value of the number of Screens in landscape mode, i.e. if you have 2 screens, enter 2. Default is 1.

    scale_h & scale_w : you can modify how much of the screen the GUI should occupy. The default is .9 and .8, respectively.

    img_scale : if you want to make the plot of the frame larger, consider changing this to .008 or more. Be careful though, too large and you will not see the buttons fully!

    Examples
    --------
    >>> deeplabcut.refine_labels('/analysis/project/reaching-task/config.yaml', Screens=2, imag_scale=.0075)
    --------

    """

    import os
    from pathlib import Path

    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if not multianimal and not cfg.get("multianimalproject", False):
        from deeplabcut.gui import refinement

        refinement.show(config)
    else:
        from deeplabcut.gui import multiple_individuals_refinement_toolbox

        multiple_individuals_refinement_toolbox.show(config)

    os.chdir(startpath)


class ExtractOutlierFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(ExtractOutlierFrames, self).__init__(root, parent, h1_description)
        self.filelist = []

        self.set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def set_page(self):

        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)

        self._generate_multianimal_options(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(
            _create_label_widget("Frame extraction options", "font:bold")
        )
        self.layout_extraction_options = _create_horizontal_layout()
        self._generate_layout_extraction_options(self.layout_extraction_options)
        self.main_layout.addLayout(self.layout_extraction_options)

        self.extract_outlierframes_button = QtWidgets.QPushButton("Extract frames")
        self.extract_outlierframes_button.clicked.connect(self.extract_outlier_frames)
        self.extract_outlierframes_button.setMinimumWidth(150)

        self.label_outliers_button = QtWidgets.QPushButton("Labeling GUI")
        self.label_outliers_button.setEnabled(False)
        self.label_outliers_button.clicked.connect(self.launch_refinement_gui)
        self.label_outliers_button.setMinimumWidth(150)

        self.merge_data_button = QtWidgets.QPushButton("Merge data")
        self.merge_data_button.setEnabled(False)
        self.merge_data_button.clicked.connect(self.merge_dataset)
        self.merge_data_button.setMinimumWidth(150)

        self.main_layout.addWidget(
            self.extract_outlierframes_button, alignment=Qt.AlignRight
        )
        self.main_layout.addWidget(self.label_outliers_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.merge_data_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

    def _generate_multianimal_options(self, layout):
        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_widget = QtWidgets.QComboBox()
        self.tracker_type_widget.addItems(DLC_Params.TRACKERS)
        self.tracker_type_widget.currentTextChanged.connect(self.update_tracker_type)

        layout.addWidget(opt_text)
        layout.addWidget(self.tracker_type_widget)
        if not self.root.is_multianimal:
            opt_text.hide()
            self.tracker_type_widget.hide()

    def _generate_layout_extraction_options(self, layout):

        opt_text = QtWidgets.QLabel("Specify the algorithm")
        self.outlier_algorithm_widget = QtWidgets.QComboBox()
        self.outlier_algorithm_widget.addItems(DLC_Params.OUTLIER_EXTRACTION_ALGORITHMS)
        self.outlier_algorithm_widget.currentTextChanged.connect(
            self.update_outlier_algorithm
        )

        layout.addWidget(opt_text)
        layout.addWidget(self.outlier_algorithm_widget)

    def update_tracker_type(self, method):
        self.root.logger.info(f"Using {method.upper()} tracker")

    def update_outlier_algorithm(self, algorithm):
        self.root.logger.info(
            f"Using {algorithm.upper()} algorithm for frame extraction"
        )

    def extract_outlier_frames(self):
        self.launch_refinement_gui.setEnabled(True)

        config = self.root.config
        shuffle = self.root.shuffle_value
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
        videos: {videos},
        videotype: {videotype},
        outlier algorithm: {outlieralgorithm},
        track method: {track_method}
        """
        )
        deeplabcut.extract_outlier_frames(
            config=config,
            videos=videos,
            videotype=videotype,
            shuffle=shuffle,
            outlieralgorithm=outlieralgorithm,
            track_method=track_method,
        )

    def launch_refinement_gui(self):
        self.merge_data_button.setEnabled(True)
        # TODO: The refinement GUI is not ported to PyQT yet.
        raise NotImplementedError
        refine_labels(self.root.config)

    def merge_dataset(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(
            "Make sure that you have refined all the labels before merging the dataset.If you merge the dataset, you need to re-create the training dataset before you start the training. Are you ready to merge the dataset?"
        )
        msg.setWindowTitle("Warning")
        msg.setWindowIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = msg.exec_()
        if result == QtWidgets.QMessageBox.Yes:
            deeplabcut.merge_datasets(self.config, forceiterate=None)
