import os

from PySide2.QtWidgets import QWidget, QMessageBox
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon

from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path

from components import (
    DefaultTab,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_horizontal_layout,
    _create_label_widget,
)
from widgets import ConfigEditor


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

    startpath = os.getcwd()
    wd = Path(config).resolve().parents[0]
    os.chdir(str(wd))
    cfg = auxiliaryfunctions.read_config(config)
    if not multianimal and not cfg.get("multianimalproject", False):
        from deeplabcut.gui import refinement

        refinement.show(config)
    else:  # loading multianimal labeling GUI
        from deeplabcut.gui import multiple_individuals_refinement_toolbox

        multiple_individuals_refinement_toolbox.show(config)

    os.chdir(startpath)


class RefineLabels(DefaultTab):
    # TODO: Add "run tracking" button + function
    def __init__(self, root, parent, h1_description):
        super(RefineLabels, self).__init__(root, parent, h1_description)
        # variable initilization

        # TODO: rename this to private -> _set_page in all tabs
        self.set_page()

    @property
    def files(self):
        self.video_selection_widget.files

    @property
    def inference_cfg_path(self):
        return os.path.join(
            self.root.cfg["project_path"],
            auxiliaryfunctions.get_model_folder(
                self.root.cfg["TrainingFraction"][0], #NOTE: trainingsetindex hardcoded!
                int(self.root.shuffle_value),
                self.root.cfg,
            ),
            "test",
            "inference_cfg.yaml",
        )

    def set_page(self):

        # TODO: Multi video select.... have to change to single video!
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.container_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        self.layout_refinement_settings = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_refinement(self.layout_refinement_settings)
        self.container_layout.addLayout(self.layout_refinement_settings)

        self.layout_filtering_settings = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_filtering(self.layout_filtering_settings)
        self.container_layout.addLayout(self.layout_filtering_settings)

        self.main_layout.addLayout(self.container_layout)

        self.edit_inferencecfg_btn = QtWidgets.QPushButton("Edit inference_cfg.yaml")
        self.edit_inferencecfg_btn.setMinimumWidth(150)
        self.edit_inferencecfg_btn.clicked.connect(self.open_inferencecfg_editor)

        self.filter_tracks_button = QtWidgets.QPushButton("Filter tracks ( + .csv)")
        self.filter_tracks_button.setMinimumWidth(150)
        self.filter_tracks_button.clicked.connect(self.filter_tracks)

        self.launch_button = QtWidgets.QPushButton("Launch refinement GUI")
        self.launch_button.setMinimumWidth(150)
        self.launch_button.clicked.connect(self.refine_labels)

        self.merge_button = QtWidgets.QPushButton("Merge dataset")
        self.merge_button.setMinimumWidth(150)
        self.merge_button.clicked.connect(self.merge_dataset)
        self.merge_button.setEnabled(False)

        self.main_layout.addWidget(self.edit_inferencecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.launch_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.filter_tracks_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.merge_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        shuffle_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QtWidgets.QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(self.root.shuffle_value)
        self.shuffle.valueChanged.connect(self.root.update_shuffle)

        # Num animals
        num_animals_text = QtWidgets.QLabel("Number of animals in video")
        self.num_animals_in_videos = QtWidgets.QSpinBox()
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.valueChanged.connect(self.log_num_animals)

        layout.addWidget(shuffle_text)
        layout.addWidget(self.shuffle)
        layout.addWidget(num_animals_text)
        layout.addWidget(self.num_animals_in_videos)
        
    def _generate_layout_refinement(self, layout):

        section_title = _create_label_widget(
            "Refinement Settings", "font:bold", (0, 50, 0, 0)
        )

        # Min swap length
        swap_length_label = QtWidgets.QLabel("Min swap length to highlight")
        self.swap_length_widget = QtWidgets.QSpinBox()
        self.swap_length_widget.setValue(2)
        self.swap_length_widget.setMinimumWidth(150)
        self.swap_length_widget.valueChanged.connect(self.log_swap_length)

        # Max gap to fill
        max_gap_label = QtWidgets.QLabel("Max gap of missing data to fill")
        self.max_gap_widget = QtWidgets.QSpinBox()
        self.max_gap_widget.setValue(5)
        self.max_gap_widget.setMinimumWidth(150)
        self.max_gap_widget.valueChanged.connect(self.log_max_gap)

        # Trail length
        trail_length_label = QtWidgets.QLabel("Visualization trail length")
        self.trail_length_widget = QtWidgets.QSpinBox()
        self.trail_length_widget.setValue(20)
        self.trail_length_widget.setMinimumWidth(150)
        self.trail_length_widget.valueChanged.connect(self.log_trail_length)

        layout.addWidget(section_title, 0, 0)
        layout.addWidget(swap_length_label, 1, 0)
        layout.addWidget(self.swap_length_widget, 1, 1)
        layout.addWidget(max_gap_label, 2, 0)
        layout.addWidget(self.max_gap_widget, 2, 1)
        layout.addWidget(trail_length_label, 3, 0)
        layout.addWidget(self.trail_length_widget, 3, 1)

    def _generate_layout_filtering(self, layout):

        section_title = _create_label_widget("Filtering", "font:bold", (0, 50, 0, 0))

        # Filter type
        filter_label = QtWidgets.QLabel("Filter type")
        self.filter_type_widget = QtWidgets.QComboBox()
        self.filter_type_widget.setMinimumWidth(150)
        options = ["median"]
        self.filter_type_widget.addItems(options)
        self.filter_type_widget.currentTextChanged.connect(self.log_filter_type)

        # Filter window length
        window_length_label = QtWidgets.QLabel("Window length")
        self.window_length_widget = QtWidgets.QSpinBox()
        self.window_length_widget.setValue(5)
        self.window_length_widget.setMinimumWidth(150)
        self.window_length_widget.valueChanged.connect(self.log_window_length)

        layout.addWidget(section_title, 0, 0)
        layout.addWidget(filter_label, 1, 0)
        layout.addWidget(self.filter_type_widget, 1, 1)
        layout.addWidget(window_length_label, 2, 0)
        layout.addWidget(self.window_length_widget, 2, 1)

    def log_swap_length(self, value):
        self.root.logger.info(f"Swap length set to {value}")

    def log_max_gap(self, value):
        self.root.logger.info(f"Max gap size of missing data to fill set to {value}")

    def log_trail_length(self, value):
        self.root.logger.info(f"Visualization trail length set to {value}")

    def log_filter_type(self, filter_type):
        self.root.logger.info(f"Filter type set to {filter_type.upper()}")

    def log_window_length(self, window_length):
        self.root.logger.info(f"Window length set to {window_length}")

    def log_num_animals(self, num_animals):
        self.root.logger.info(f"Number of animals in video set to {num_animals}")

    def open_inferencecfg_editor(self):
        editor = ConfigEditor(self.inference_cfg_path)
        editor.show()

    def filter_tracks(self):
        #TODO: 
        raise NotImplementedError

    def merge_dataset(self):
        # TODO: make sure this is correct

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(
            "Make sure that you have refined all the labels before merging the dataset.If you merge the dataset, you need to re-create the training dataset before you start the training. Are you ready to merge the dataset?"
        )
        msg.setWindowTitle("Warning")
        msg.setWindowIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = msg.exec_()
        if result == QMessageBox.Yes:
            pass
            # TODO: finish -->
            # notebook = self.GetParent()
            # notebook.SetSelection(4)
            # deeplabcut.merge_datasets(self.config, forceiterate=None)

    def refine_labels(self):
        self.merge_button.setEnabled(True)
        # TODO: finish refine_labels part
        # deeplabcut.refine_labels(self.config)
