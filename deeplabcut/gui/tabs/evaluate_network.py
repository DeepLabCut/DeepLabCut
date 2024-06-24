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
from __future__ import annotations

import os
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from pathlib import Path
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.gui.displays.selected_shuffle_display import SelectedShuffleDisplay
from deeplabcut.gui.components import (
    BodypartListWidget,
    DefaultTab,
    ShuffleSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
from deeplabcut.gui.widgets import ConfigEditor, launch_napari
from deeplabcut.utils import auxiliaryfunctions


class GridCanvas(QtWidgets.QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure()
        self.figure.patch.set_facecolor("None")
        self.grid = self.figure.add_gridspec(3, 3)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        for image_path, gridspec in zip(image_paths[:9], self.grid):
            ax = self.figure.add_subplot(gridspec)
            ax.set_axis_off()
            img = mpimg.imread(image_path)
            ax.imshow(img)


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

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

        self.root.engine_change.connect(self._on_engine_change)
        self._on_engine_change(self.root.engine)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(deeplabcut.evaluate_network.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self, layout):
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)
        self.shuffle_display = SelectedShuffleDisplay(self.root, row_margin=0)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)
        layout.addWidget(self.shuffle_display)

    def open_inferencecfg_editor(self):
        editor = ConfigEditor(self.root.inference_cfg_path)
        editor.show()

    def plot_maps(self):
        shuffle = self.root.shuffle_value
        config = self.root.config
        deeplabcut.extract_save_all_maps(config, shuffle=shuffle, Indices=[0, 1, 2])

        # Display all images
        dest_folder = os.path.join(
            self.root.project_folder,
            str(
                auxiliaryfunctions.get_evaluation_folder(
                    self.root.cfg["TrainingFraction"][0], shuffle, self.root.cfg
                )
            ),
            "maps",
        )
        image_paths = [
            os.path.join(dest_folder, file)
            for file in os.listdir(dest_folder)
            if file.endswith(".png")
        ]
        canvas = GridCanvas(image_paths, parent=self)
        canvas.show()

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
        if Qt.CheckState(state) == Qt.Checked:
            self.root.logger.info("Plot scoremaps ENABLED")
        else:
            self.root.logger.info("Plot predictions DISABLED")

    def update_plot_predictions(self, s):
        if Qt.CheckState(s) == Qt.Checked:
            self.root.logger.info("Plot predictions ENABLED")
        else:
            self.root.logger.info("Plot predictions DISABLED")

    def update_bodypart_choice(self, s):
        if Qt.CheckState(s) == Qt.Checked:
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
        shuffle = self.root.shuffle_value
        plotting = self.plot_predictions.isChecked()

        bodyparts_to_use = "all"
        if (
            len(self.root.all_bodyparts)
            != len(self.bodyparts_list_widget.selected_bodyparts)
        ) and not self.use_all_bodyparts.isChecked():
            bodyparts_to_use = self.bodyparts_list_widget.selected_bodyparts

        deeplabcut.evaluate_network(
            config,
            Shuffles=[shuffle],
            plotting=plotting,
            show_errors=True,
            comparisonbodyparts=bodyparts_to_use,
        )

        if plotting:
            project_cfg = self.root.cfg
            eval_folder = auxiliaryfunctions.get_evaluation_folder(
                trainFraction=project_cfg["TrainingFraction"][0],
                shuffle=shuffle,
                cfg=project_cfg,
            )
            scorer, _ = auxiliaryfunctions.get_scorer_name(
                cfg=project_cfg,
                shuffle=shuffle,
                trainFraction=project_cfg["TrainingFraction"][0],
            )

            image_dir = (
                Path(self.root.project_folder)
                / eval_folder
                / f"LabeledImages_{scorer}"
            )
            labeled_images = [str(p) for p in image_dir.rglob("*.png")]
            if len(labeled_images) > 0:
                _ = launch_napari(image_dir)

    @Slot(Engine)
    def _on_engine_change(self, engine: Engine) -> None:
        if engine == Engine.PYTORCH:
            self.opt_button.hide()
            return

        self.opt_button.show()
