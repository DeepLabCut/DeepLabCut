from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QComboBox, QSpinBox, QButtonGroup

from deeplabcut.generate_training_dataset import extract_frames

from components import (
    _create_grid_layout,
    _create_horizontal_layout,
    DefaultTab,
    _create_label_widget,
    _create_vertical_layout,
)


class ExtractFrames(DefaultTab):
    # NOTE: The layout of this tab is behaving wirdly 
    # (expands according to window height) but I cannot tell why.
    def __init__(self, root, parent, h1_description):
        super(ExtractFrames, self).__init__(root, parent, h1_description)

        self.set_page()

    def set_page(self):

        self.main_layout.addWidget(
            _create_label_widget("Attributes", "font:bold")
        )
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.extract_frames)

        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):

        # Extraction method
        ext_method_label = QtWidgets.QLabel("Extraction method")

        self.extraction_method_widget = QtWidgets.QComboBox()
        self.extraction_method_widget.setMinimumWidth(150)
        self.extraction_method_widget.setMinimumHeight(30)
        options = ["automatic", "manual"]
        self.extraction_method_widget.addItems(options)
        self.extraction_method_widget.currentTextChanged.connect(
            self.log_extraction_method
        )


        # User feedback
        self.user_feedback_checkbox = QtWidgets.QCheckBox("User feedback")
        self.user_feedback_checkbox.setCheckState(Qt.Unchecked)
        self.user_feedback_checkbox.stateChanged.connect(self.log_user_feedback_choice)

        # Frame extraction algorithm
        ext_algo_label = QtWidgets.QLabel("Extraction algorithm")

        self.extraction_algorithm_widget = QtWidgets.QComboBox()
        self.extraction_algorithm_widget.setMinimumWidth(150)
        self.extraction_algorithm_widget.setMinimumHeight(30)
        options = ["kmeans", "uniform"]
        self.extraction_algorithm_widget.addItems(options)
        self.extraction_algorithm_widget.currentTextChanged.connect(
            self.log_extraction_algorithm
        )

        # Frame cropping
        frame_crop_label = QtWidgets.QLabel("Frame Cropping")

        self.frame_cropping_widget = QtWidgets.QComboBox()
        self.frame_cropping_widget.setMinimumWidth(150)
        self.frame_cropping_widget.setMinimumHeight(30)
        options = ["disabled", "read from config", "GUI"]
        self.frame_cropping_widget.addItems(options)
        self.frame_cropping_widget.currentTextChanged.connect(
            self.log_frame_cropping_choice
        )

        # Use openCV
        self.use_openCV_checkbox = QtWidgets.QCheckBox("Use openCV")
        self.use_openCV_checkbox.setCheckState(Qt.Checked)
        self.use_openCV_checkbox.stateChanged.connect(self.update_opencv_choice)
        self.use_openCV_checkbox.setToolTip(
            "Recommended. Uses openCV for managing videos instead of moviepy (legacy)."
        )


        # Cluster step
        cluster_step_label = QtWidgets.QLabel("Cluster step")

        self.cluster_step_widget = QSpinBox()
        self.cluster_step_widget.setValue(25)
        # self.cluster_step_widget.setMinimumWidth(100)
        self.cluster_step_widget.setMinimumHeight(30)

        # GUI Slider width
        gui_slider_label = QtWidgets.QLabel("GUI slider width")

        self.slider_width_widget = QSpinBox()
        self.slider_width_widget.setValue(25)
        self.slider_width_widget.setMinimumWidth(100)
        self.slider_width_widget.setMinimumHeight(30)
        self.slider_width_widget.setEnabled(False)

        # 1st attributes line
        layout.addWidget(self.user_feedback_checkbox, 0, 0)
        layout.addWidget(self.use_openCV_checkbox, 0, 1)

        # 2nd attributes line
        layout.addWidget(ext_method_label, 1, 0)
        layout.addWidget(self.extraction_method_widget, 1, 1)
        layout.addWidget(gui_slider_label, 1, 2)
        layout.addWidget(self.slider_width_widget, 1, 3)

        # 3rd attributes line
        layout.addWidget(ext_algo_label, 2, 0)
        layout.addWidget(self.extraction_algorithm_widget , 2, 1)
        layout.addWidget(cluster_step_label, 2, 2)
        layout.addWidget(self.cluster_step_widget, 2, 3)

        # 4th attributes line
        layout.addWidget(frame_crop_label, 3, 0)
        layout.addWidget(self.frame_cropping_widget, 3, 1)

    def log_user_feedback_choice(self, state):
        if state == Qt.Checked:
            self.root.logger.info("User feedback ENABLED")
        else:
            self.root.logger.info("User feedback DISABLED")

    def log_extraction_algorithm(self, extraction_algorithm):
        self.root.logger.info(f"Extraction method set to {extraction_algorithm}")

    def log_extraction_method(self, extraction_method):
        self.root.logger.info(f"Extraction method set to {extraction_method}")
        if extraction_method == "manual":
            self.extraction_algorithm_widget.setEnabled(False)
            self.cluster_step_widget.setEnabled(False)
            self.slider_width_widget.setEnabled(True)
        else:
            self.extraction_algorithm_widget.setEnabled(True)
            self.cluster_step_widget.setEnabled(True)
            self.slider_width_widget.setEnabled(False)


    def log_frame_cropping_choice(self, cropping_option):
        self.root.logger.info(f"Cropping set to '{cropping_option}'")

    def update_feedback_choice(self, s):
        if s == Qt.Checked:
            self.feedback = True
            self.logger.info("Enabling user feedback.")
        else:
            self.feedback = False
            self.logger.info("Disabling user feedback.")

    def update_opencv_choice(self, s):
        if s == Qt.Checked:
            self.logger.info("Use openCV enabled.")
        else:
            self.logger.info("Use openCV disabled. Using moviepy..")

    def extract_frames(self):

        config = self.root.config
        mode = self.extraction_method_widget.currentText()
        algo = self.extraction_algorithm_widget.currentText()
        userfeedback = self.user_feedback_checkbox.checkState() == Qt.Checked
        opencv = self.use_openCV_checkbox.checkState() == Qt.Checked
        clusterstep = self.cluster_step_widget.value()
        slider_width = self.slider_width_widget.value()

        crop = False # default value
        if self.frame_cropping_widget.currentText() == "GUI":
            # TODO: Plug GUI cropping
            raise NotImplementedError
        elif self.frame_cropping_widget.currentText() == "read from config":
            crop = True

        extract_frames(
            config,
            mode,
            algo,
            crop=crop,
            userfeedback=userfeedback,
            cluster_step=clusterstep,
            cluster_resizewidth=30,
            cluster_color=False,
            opencv=opencv,
            slider_width=slider_width,
        )

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Frames were successfully extracted, for the videos of interest.")

        msg.setWindowTitle("Info")
        msg.setWindowIcon(QtWidgets.QMessageBox.Information)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
