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
import os
from dataclasses import dataclass

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import deeplabcut.compat as compat
from deeplabcut.core.engine import Engine
from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.widgets import ConfigEditor


@dataclass
class IntTrainAttribute:
    label: str
    fn_key: str
    default: int
    min: int
    max: int


class TrainNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description, engine: Engine = Engine.TF):
        super(TrainNetwork, self).__init__(root, parent, h1_description)
        self.train_attributes = get_train_attributes(engine=engine)
        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        self.edit_posecfg_btn = QtWidgets.QPushButton("Edit pose_cfg.yaml")
        self.edit_posecfg_btn.setMinimumWidth(150)
        self.edit_posecfg_btn.clicked.connect(self.open_posecfg_editor)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.train_network)

        self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(deeplabcut.train_network.__doc__, self)
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
        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)

        # Other parameters
        self.attribute_spin_boxes = {}
        for i, attribute in enumerate(self.train_attributes):
            label = QtWidgets.QLabel(attribute.label)
            spin_box = QtWidgets.QSpinBox()
            spin_box.setMinimum(attribute.min)
            spin_box.setMaximum(attribute.max)
            spin_box.setValue(attribute.default)
            spin_box.valueChanged.connect(
                lambda new_val: self.log_attribute_change(attribute, new_val)
            )
            self.attribute_spin_boxes[attribute.fn_key] = spin_box
            layout.addWidget(label, 0, 2 * (i + 1))
            layout.addWidget(spin_box, 0, 2 * (i + 1) + 1)

    def log_attribute_change(self, attribute: IntTrainAttribute, value: int) -> None:
        self.root.logger.info(f"{attribute.label} set to {value}")

    def open_posecfg_editor(self):
        editor = ConfigEditor(self.root.pose_cfg_path)
        editor.show()

    def train_network(self):
        config = self.root.config
        shuffle = int(self.shuffle.value())
        kwargs = dict(gputouse=None, autotune=False)
        for k, spin_box in self.attribute_spin_boxes.items():
            kwargs[k] = int(spin_box.value())

        compat.train_network(config, shuffle, **kwargs)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The network is now trained and ready to evaluate.")
        msg.setInformativeText(
            "Use the function 'evaluate_network' to evaluate the network."
        )

        msg.setWindowTitle("Info")
        msg.setMinimumWidth(900)
        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()


def get_train_attributes(engine: Engine) -> list[IntTrainAttribute]:
    if engine == Engine.TF:
        return [
            IntTrainAttribute(
                label="Display iterations",
                fn_key="displayiters",
                default=1000,
                min=1,
                max=1000,
            ),
            IntTrainAttribute(
                label="Save iterations",
                fn_key="saveiters",
                default=50_000,
                min=1,
                max=50_000,
            ),
            IntTrainAttribute(
                label="Maximum iterations",
                fn_key="maxiters",
                default=100_000,
                min=1,
                max=1_030_000,
            ),
            IntTrainAttribute(
                label="Number of snapshots to keep",
                fn_key="max_snapshots_to_keep",
                default=5,
                min=1,
                max=100,
            ),
        ]
    elif engine == Engine.PYTORCH:
        return[
            IntTrainAttribute(
                label="Display iterations",
                fn_key="display_iters",
                default=1_000,
                min=1,
                max=100_000,
            ),
            IntTrainAttribute(
                label="Save epochs",
                fn_key="save_epochs",
                default=50,
                min=1,
                max=250,
            ),
            IntTrainAttribute(
                label="Maximum epochs",
                fn_key="epochs",
                default=200,
                min=1,
                max=1000,
            ),
            # IntTrainAttribute(  # FIXME: Implement
            #     label="Number of snapshots to keep",
            #     fn_key="max_snapshots_to_keep",
            #     default=5,
            #     min=1,
            #     max=100,
            # ),
        ]

    raise NotImplementedError(f"Unknown engine: {engine}")
