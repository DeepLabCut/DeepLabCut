import os

from PySide2.QtWidgets import QSpinBox
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from components import (
    BodypartListWidget,
    DefaultTab,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
from widgets import ConfigEditor


class EvaluateNetwork(DefaultTab):
    def __init__(self, root, parent, tab_heading):
        super(EvaluateNetwork, self).__init__(root, parent, tab_heading)

        self.method = "automatic"

        self.plot_choice = False
        self.plot_scoremaps = False

        self.bodyparts_to_use = self.root.all_bodyparts

        self.set_page()

    def set_page(self):

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget("")) #dummy text
        self.layout_additional_attributes = _create_vertical_layout()
        self._generate_additional_attributes(self.layout_additional_attributes)

        self.main_layout.addLayout(self.layout_additional_attributes)

        self.ev_nw_button = QtWidgets.QPushButton("Evaluate Network")
        self.ev_nw_button.setMinimumWidth(200)
        self.ev_nw_button.setContentsMargins(0, 80, 40, 40)
        self.ev_nw_button.clicked.connect(self.evaluate_network)

        self.opt_button = QtWidgets.QPushButton("Plot 3 test maps")
        self.opt_button.setMinimumWidth(200)
        self.opt_button.setContentsMargins(0, 80, 40, 40)
        self.opt_button.clicked.connect(self.plot_maps)

        self.edit_posecfg_btn = QtWidgets.QPushButton("Edit inference_cfg.yaml")
        self.edit_posecfg_btn.setMinimumWidth(200)
        self.edit_posecfg_btn.setContentsMargins(0, 80, 40, 40)
        self.edit_posecfg_btn.clicked.connect(self.open_inferencecfg_editor)

        self.main_layout.addLayout(self.layout_attributes)

        if self.root.is_multianimal:
            self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)

        self.main_layout.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)


    @property
    def inference_cfg_path(self):
        return os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.GetModelFolder(
                self.cfg["TrainingFraction"][int(self.trainingsetidx.value())],
                int(self.shuffle.value()),
                self.cfg,
            ),
            "test",
            "inference_cfg.yaml",
        )

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(self.root.shuffle_value)
        self.shuffle.setMinimumHeight(30)
        self.shuffle.valueChanged.connect(self.root.update_shuffle)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingsetidx = QSpinBox()
        self.trainingsetidx.setMaximum(100)
        self.trainingsetidx.setValue(0)
        self.trainingsetidx.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.trainingsetidx)

    def open_inferencecfg_editor(self):
        editor = ConfigEditor(self.inference_cfg_path)
        editor.show()

    def plot_maps(self):
        shuffle = self.root.shuffle_value
        config = self.root.config
        deeplabcut.extract_save_all_maps(
            config, shuffle=shuffle, Indices=[0, 1, 5]
        )

    def _generate_additional_attributes(self, layout):

        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        self.plot_predictions = QtWidgets.QCheckBox(
            "Plot predictions (as in standard DLC projects)"
        )
        self.plot_predictions.stateChanged.connect(self.update_plot_predictions)

        tmp_layout.addWidget(self.plot_predictions)

        self.use_all_bodyparts = QtWidgets.QCheckBox("Compare all bodyparts")
        self.use_all_bodyparts.stateChanged.connect(self.update_bodypart_choice)
        self.use_all_bodyparts.setCheckState(Qt.Checked)

        tmp_layout.addWidget(self.use_all_bodyparts)
        layout.addLayout(tmp_layout)

        self.bodyparts_list_widget = BodypartListWidget(
            root=self.root, parent=self,
        )
        self.bodyparts_list_widget.setMaximumWidth(600)
        self.bodyparts_list_widget.setMaximumHeight(500)
        layout.addWidget(self.bodyparts_list_widget, alignment=Qt.AlignLeft)

    def update_map_choice(self, state):
        if state == Qt.Checked:
            self.logger.info("Plot scoremaps ENABLED")
        else:
            self.logger.info("Plot predictions DISABLED")

    def update_plot_predictions(self, s):
        if s == Qt.Checked:
            self.plot_choice = True
            self.logger.info("Plot predictions ENABLED")
        else:
            self.plot_choice = False
            self.logger.info("Plot predictions DISABLED")

    def update_bodypart_choice(self, s):
        if s == Qt.Checked:
            self.bodyparts_list_widget.setEnabled(False)
            self.logger.info("Use all bodyparts")
        else:
            self.bodyparts_list_widget.setEnabled(True)
            self.logger.info(
                f"Use selected bodyparts only: {self.bodyparts_list_widget.selected_bodyparts}"
            )

    def evaluate_network(self):

        config = self.root.config
        trainingsetindex = self.trainingsetidx.value()

        Shuffles = [self.root.shuffle_value]
        plotting = self.plot_choice

        bodyparts_to_use = "all"
        if (
            len(self.root.all_bodyparts)
            != len(self.bodyparts_list_widget.selected_bodyparts)
        ) and self.use_all_bodyparts.checkState() == False:
            bodyparts_to_use = self.bodyparts_list_widget.selected_bodyparts

        deeplabcut.evaluate_network(
            config,
            Shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=True,
            comparisonbodyparts=bodyparts_to_use,
        )