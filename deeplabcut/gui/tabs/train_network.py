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
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.widgets import ConfigEditor

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


class TrainNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)

        # use the default pose_cfg file for default values
        default_pose_cfg_path = os.path.join(
            Path(deeplabcut.__file__).parent, "pose_cfg.yaml"
        )
        pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        self.display_iters = str(pose_cfg["display_iters"])
        self.save_iters = str(pose_cfg["save_iters"])
        self.max_iters = str(pose_cfg["multi_step"][-1][-1])

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

    def _generate_layout_attributes(self, layout):
        # Shuffle
        shuffle_label = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Display iterations")
        self.display_iters_spin = QtWidgets.QSpinBox()
        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(int(self.max_iters))
        self.display_iters_spin.setValue(1000)
        self.display_iters_spin.valueChanged.connect(self.log_display_iters)

        # Save iterations
        saveiters_label = QtWidgets.QLabel("Save iterations")
        self.save_iters_spin = QtWidgets.QSpinBox()
        self.save_iters_spin.setMinimum(1)
        self.save_iters_spin.setMaximum(int(self.max_iters))
        self.save_iters_spin.setValue(50000)
        self.save_iters_spin.valueChanged.connect(self.log_save_iters)

        # Max iterations
        maxiters_label = QtWidgets.QLabel("Maximum iterations")
        self.max_iters_spin = QtWidgets.QSpinBox()
        self.max_iters_spin.setMinimum(1)
        self.max_iters_spin.setMaximum(int(self.max_iters))
        self.max_iters_spin.setValue(100000)
        self.max_iters_spin.valueChanged.connect(self.log_max_iters)

        # Max number snapshots to keep
        snapkeep_label = QtWidgets.QLabel("Number of snapshots to keep")
        self.snapshots = QtWidgets.QSpinBox()
        self.snapshots.setMinimum(1)
        self.snapshots.setMaximum(100)
        self.snapshots.setValue(5)
        self.snapshots.valueChanged.connect(self.log_snapshots)

        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(dispiters_label, 0, 2)
        layout.addWidget(self.display_iters_spin, 0, 3)
        layout.addWidget(saveiters_label, 0, 4)
        layout.addWidget(self.save_iters_spin, 0, 5)
        layout.addWidget(maxiters_label, 0, 6)
        layout.addWidget(self.max_iters_spin, 0, 7)
        layout.addWidget(snapkeep_label, 0, 8)
        layout.addWidget(self.snapshots, 0, 9)
        # layout.addWidget()

    def log_display_iters(self, value):
        self.root.logger.info(f"Display iters set to {value}")

    def log_save_iters(self, value):
        self.root.logger.info(f"Save iters set to {value}")

    def log_max_iters(self, value):
        self.root.logger.info(f"Max iters set to {value}")

    def log_snapshots(self, value):
        self.root.logger.info(f"Max snapshots to keep set to {value}")

    def open_posecfg_editor(self):
        editor = ConfigEditor(self.root.pose_cfg_path)
        editor.show()

    def train_network(self):

        config = self.root.config
        shuffle = int(self.shuffle.value())
        max_snapshots_to_keep = int(self.snapshots.value())
        displayiters = int(self.display_iters_spin.value())
        saveiters = int(self.save_iters_spin.value())
        maxiters = int(self.max_iters_spin.value())

        deeplabcut.train_network(
            config,
            shuffle,
            gputouse=None,
            max_snapshots_to_keep=max_snapshots_to_keep,
            autotune=None,
            displayiters=displayiters,
            saveiters=saveiters,
            maxiters=maxiters,
        )
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
