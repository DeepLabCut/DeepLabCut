from PySide2 import QtWidgets
from PySide2.QtCore import Qt

from deeplabcut_gui.components import (
    BodypartListWidget,
    DefaultTab,
    ShuffleSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
from deeplabcut_gui.widgets import ConfigEditor

import deeplabcut


class EvaluateNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(EvaluateNetwork, self).__init__(root, parent, h1_description)

        self.bodyparts_to_use = self.root.all_bodyparts

        self._set_page()

    def _set_page(self):

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_horizontal_layout()
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy text
        self.layout_additional_attributes = _create_vertical_layout()
        self._generate_additional_attributes(self.layout_additional_attributes)
        self.main_layout.addLayout(self.layout_additional_attributes)

        self.ev_nw_button = QtWidgets.QPushButton("Evaluate Network")
        self.ev_nw_button.setMinimumWidth(150)
        self.ev_nw_button.clicked.connect(self.evaluate_network)

        self.opt_button = QtWidgets.QPushButton("Plot 3 test maps")
        self.opt_button.setMinimumWidth(150)
        self.opt_button.clicked.connect(self.plot_maps)

        self.edit_inferencecfg_btn = QtWidgets.QPushButton("Edit inference_cfg.yaml")
        self.edit_inferencecfg_btn.setMinimumWidth(150)
        self.edit_inferencecfg_btn.clicked.connect(self.open_inferencecfg_editor)

        if self.root.is_multianimal:
            self.main_layout.addWidget(
                self.edit_inferencecfg_btn, alignment=Qt.AlignRight
            )

        self.main_layout.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

    def open_inferencecfg_editor(self):
        editor = ConfigEditor(self.root.inference_cfg_path)
        editor.show()

    def plot_maps(self):
        shuffle = self.root.shuffle_value
        config = self.root.config
        deeplabcut.extract_save_all_maps(config, shuffle=shuffle, Indices=[0, 1, 5])

    def _generate_additional_attributes(self, layout):

        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        self.plot_predictions = QtWidgets.QCheckBox(
            "Plot predictions (as in standard DLC projects)"
        )
        self.plot_predictions.stateChanged.connect(self.update_plot_predictions)

        tmp_layout.addWidget(self.plot_predictions)

        self.bodyparts_list_widget = BodypartListWidget(root=self.root, parent=self)
        self.use_all_bodyparts = QtWidgets.QCheckBox("Compare all bodyparts")
        self.use_all_bodyparts.stateChanged.connect(self.update_bodypart_choice)
        self.use_all_bodyparts.setCheckState(Qt.Checked)

        tmp_layout.addWidget(self.use_all_bodyparts)
        layout.addLayout(tmp_layout)

        layout.addWidget(self.bodyparts_list_widget, alignment=Qt.AlignLeft)

    def update_map_choice(self, state):
        if state == Qt.Checked:
            self.root.logger.info("Plot scoremaps ENABLED")
        else:
            self.root.logger.info("Plot predictions DISABLED")

    def update_plot_predictions(self, s):
        if s == Qt.Checked:
            self.root.logger.info("Plot predictions ENABLED")
        else:
            self.root.logger.info("Plot predictions DISABLED")

    def update_bodypart_choice(self, s):
        if s == Qt.Checked:
            self.bodyparts_list_widget.setEnabled(False)
            self.bodyparts_list_widget.hide()
            self.root.logger.info("Use all bodyparts")
        else:
            self.bodyparts_list_widget.setEnabled(True)
            self.bodyparts_list_widget.show()
            self.root.logger.info(
                f"Use selected bodyparts only: {self.bodyparts_list_widget.selected_bodyparts}"
            )

    def evaluate_network(self):

        config = self.root.config

        Shuffles = [self.root.shuffle_value]
        plotting = self.plot_predictions.checkState() == Qt.Checked

        bodyparts_to_use = "all"
        if (
            len(self.root.all_bodyparts)
            != len(self.bodyparts_list_widget.selected_bodyparts)
        ) and self.use_all_bodyparts.checkState() == False:
            bodyparts_to_use = self.bodyparts_list_widget.selected_bodyparts

        deeplabcut.evaluate_network(
            config,
            Shuffles=Shuffles,
            plotting=plotting,
            show_errors=True,
            comparisonbodyparts=bodyparts_to_use,
        )
