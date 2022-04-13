import os
import sys
import subprocess
from pathlib import Path

from PySide2.QtWidgets import QWidget, QSpinBox, QButtonGroup
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from components import EditYamlButton, _create_horizontal_layout, _create_label_widget


class TrainNetwork(QWidget):
    def __init__(self, parent, cfg):
        super(TrainNetwork, self).__init__(parent)

        self.method = "automatic"
        self.userfeedback = False
        self.model_comparison = False
        self.pose_cfg_choice = False
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)


        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(0, 20, 10, 20)
        self.setLayout(self.main_layout)

        # use the default pose_cfg file for default values
        default_pose_cfg_path = os.path.join(
            Path(deeplabcut.__file__).parent, "pose_cfg.yaml"
        )

        pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        self.display_iters = str(pose_cfg["display_iters"])
        self.save_iters = str(pose_cfg["save_iters"])
        self.max_iters = str(pose_cfg["multi_step"][-1][-1])

        self.set_page()

    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 5. Train network")
        l1_step1.setStyleSheet("font:bold")    
        l1_step1.setContentsMargins(20, 0, 0, 10)

        self.main_layout.addWidget(l1_step1)
        self.main_layout.addWidget(separatorLine)

        layout_config = _create_horizontal_layout()
        self._generate_config_layout(layout_config)
        self.main_layout.addLayout(layout_config)

        self.main_layout.addWidget(
            _create_label_widget("Attributes", "font:bold")
        )
        tmp_layout = _create_horizontal_layout()
        self._generate_layout_attributes(tmp_layout)
        self.main_layout.addLayout(tmp_layout)

        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setContentsMargins(0, 0, 10, 0)

        self.layout_display = QtWidgets.QHBoxLayout()
        self.layout_display.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_display.setSpacing(20)
        self.layout_display.setContentsMargins(0, 0, 20, 0)

        self._display()
        self._save()
        self._max()
        self._num_of_snapshots()

        self.layout_attributes.addLayout(self.layout_display)

        self.pose_cfg_path = os.path.join(
            self.cfg["project_path"],
            auxiliaryfunctions.GetModelFolder(
                self.cfg["TrainingFraction"][int(self.trainingindex.value())],
                int(self.shuffle.value()),
                self.cfg,
            ),
            "train",
            "pose_cfg.yaml",
        )
        self.edit_posecfg_btn = EditYamlButton(
            "Edit pose_cfg.yaml", self.pose_cfg_path, parent=self)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.clicked.connect(self.train_network)
        
        self.layout_attributes.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.layout_attributes.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.main_layout.addLayout(self.layout_attributes)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = QSpinBox()
        self.shuffle.setMaximum(100)
        self.shuffle.setValue(1)
        self.shuffle.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.shuffle)

        # Trainingset index
        opt_text = QtWidgets.QLabel("Trainingset index")
        self.trainingindex = QSpinBox()
        self.trainingindex.setMaximum(100)
        self.trainingindex.setValue(0)
        self.trainingindex.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.trainingindex)

    def _generate_config_layout(self, layout):
        cfg_text = QtWidgets.QLabel("Active config file:")

        self.cfg_line = QtWidgets.QLineEdit()
        self.cfg_line.setMinimumHeight(30)
        self.cfg_line.setText(self.config)
        self.cfg_line.textChanged[str].connect(self.update_cfg)

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.setMinimumHeight(30)
        browse_button.clicked.connect(self.browse_dir)

        layout.addWidget(cfg_text)
        layout.addWidget(self.cfg_line)
        layout.addWidget(browse_button)


    def update_cfg(self):
        text = self.cfg_line.text()
        self.config = text
        self.cfg = auxiliaryfunctions.read_config(self.config)


    def browse_dir(self):
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config[0]:
            return
        self.config = config[0]
        self.cfg_line.setText(self.config)

    def _display(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Display iterations")
        self.display_iters_spin = QSpinBox()

        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(int(self.max_iters))
        self.display_iters_spin.setValue(1000)

        self.display_iters_spin.setMinimumWidth(300)
        self.display_iters_spin.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.display_iters_spin)
        self.layout_display.addLayout(l_opt)

    def _save(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Save iterations")
        self.save_iters_spin = QSpinBox()

        self.save_iters_spin.setMinimum(1)
        self.save_iters_spin.setMaximum(int(self.max_iters))
        self.save_iters_spin.setValue(50000)

        self.save_iters_spin.setMinimumWidth(300)
        self.save_iters_spin.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.save_iters_spin)
        self.layout_display.addLayout(l_opt)

    def _max(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Maximum iterations")
        self.max_iters_spin = QSpinBox()

        self.max_iters_spin.setMinimum(1)
        self.max_iters_spin.setMaximum(int(self.max_iters))
        self.max_iters_spin.setValue(1030000)
        self.max_iters_spin.setMinimumWidth(300)
        self.max_iters_spin.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.max_iters_spin)
        self.layout_display.addLayout(l_opt)

    def _num_of_snapshots(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Number of snapshots to keep")
        self.snapshots = QSpinBox()
        self.snapshots.setValue(5)
        self.snapshots.setMinimum(1)
        self.snapshots.setMaximum(100)
        self.snapshots.setMinimumWidth(300)
        self.snapshots.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.snapshots)
        self.layout_display.addLayout(l_opt)

    def train_network(self):

        shuffle = int(self.shuffle.value())
        trainingsetindex = int(self.trainingindex.value())
        max_snapshots_to_keep = int(self.snapshots.value())
        displayiters = int(self.display_iters_spin.value())
        saveiters = int(self.save_iters_spin.value())
        maxiters = int(self.max_iters_spin.value())

        deeplabcut.train_network(
            self.config,
            shuffle,
            trainingsetindex,
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
