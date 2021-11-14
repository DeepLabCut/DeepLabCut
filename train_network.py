import os
import pydoc
import subprocess
import sys
import webbrowser
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

class Train_network_page(QWidget):

    def __init__(self, parent, cfg):
        super(Train_network_page, self).__init__(parent)

        self.method = "automatic"
        self.userfeedback = False
        self.model_comparison = False
        self.pose_cfg_choice = False
        self.config = cfg

        self.inLayout = QtWidgets.QVBoxLayout(self)
        self.inLayout.setAlignment(Qt.AlignTop)
        self.inLayout.setSpacing(20)
        self.inLayout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(self.inLayout)

        # use the default pose_cfg file for default values
        default_pose_cfg_path = os.path.join(
            Path(deeplabcut.__file__).parent, "pose_cfg.yaml"
        )
        #pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        # self.display_iters = str(pose_cfg["display_iters"])
        # self.save_iters = str(pose_cfg["save_iters"])
        # self.max_iters = str(pose_cfg["multi_step"][-1][-1])

        self.display_iters = 1000
        self.save_iters = 50000
        self.max_iters = 1030000

        self.set_page()


    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 5. Train network")
        l1_step1.setContentsMargins(20, 0, 0, 10)

        self.inLayout.addWidget(l1_step1)
        self.inLayout.addWidget(separatorLine)

        layout_cfg = QtWidgets.QHBoxLayout()
        layout_cfg.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout_cfg.setSpacing(20)
        layout_cfg.setContentsMargins(20, 10, 300, 0)
        cfg_text = QtWidgets.QLabel("Select the config file")
        cfg_text.setContentsMargins(0, 0, 60, 0)

        self.cfg_line = QtWidgets.QLineEdit()
        self.cfg_line.setMaximumWidth(800)
        self.cfg_line.setMinimumWidth(600)
        self.cfg_line.setMinimumHeight(30)
        self.cfg_line.setText(self.config)
        self.cfg_line.textChanged[str].connect(self.update_cfg)

        browse_button = QtWidgets.QPushButton('Browse')
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_dir)

        layout_cfg.addWidget(cfg_text)
        layout_cfg.addWidget(self.cfg_line)
        layout_cfg.addWidget(browse_button)

        self.inLayout.addLayout(layout_cfg)

        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setSpacing(20)
        self.layout_attributes.setContentsMargins(0, 0, 40, 0)

        label = QtWidgets.QLabel('Optional Attributes')
        label.setContentsMargins(20, 20, 0, 10)
        self.layout_attributes.addWidget(label)

        self.layout_specify_edit = QtWidgets.QHBoxLayout()
        self.layout_specify_edit.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_specify_edit.setSpacing(20)
        self.layout_specify_edit.setContentsMargins(20, 0, 50, 20)

        self._specify_shuffle()
        self._specify_ts_index()
        self._edit_file()

        self.layout_display = QtWidgets.QHBoxLayout()
        self.layout_display.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_display.setSpacing(20)
        self.layout_display.setContentsMargins(20, 0, 50, 0)

        self._display()
        self._save()
        self._max()
        self._num_of_snapshots()

        self.layout_attributes.addLayout(self.layout_specify_edit)
        self.layout_attributes.addLayout(self.layout_display)

        self.ok_button = QtWidgets.QPushButton('Ok')
        self.ok_button.setContentsMargins(0, 80, 40, 40)
        #self.ok_button.clicked.connect(self.train_network)

        self.layout_attributes.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.inLayout.addLayout(self.layout_attributes)


    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):
        print('browse_dir')
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )
        if not config:
            return
        self.config = config

    def _specify_shuffle(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the shuffle")
        self.shuffles = QSpinBox()
        self.shuffles.setValue(1)
        self.shuffles.setMinimumWidth(400)
        self.shuffles.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.shuffles)
        self.layout_specify_edit.addLayout(l_opt)
    def _specify_ts_index(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the trainingset index")
        self.trainingindex = QSpinBox()
        self.trainingindex.setValue(0)
        self.trainingindex.setMinimumWidth(400)
        self.trainingindex.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.trainingindex)
        self.layout_specify_edit.addLayout(l_opt)
    def _edit_file(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Want to edit pose_cfg.yaml file?")
        self.btngroup_pose_cfg_choice = QButtonGroup()

        self.pose_cfg_choice1 = QtWidgets.QRadioButton('Yes')
        self.pose_cfg_choice1.toggled.connect(lambda: self.update_pose_cfg_choice(self.pose_cfg_choice1))

        self.pose_cfg_choice2 = QtWidgets.QRadioButton('No')
        self.pose_cfg_choice2.setChecked(True)
        self.pose_cfg_choice2.toggled.connect(lambda: self.update_pose_cfg_choice(self.pose_cfg_choice2))

        self.btngroup_pose_cfg_choice.addButton(self.pose_cfg_choice1)
        self.btngroup_pose_cfg_choice.addButton(self.pose_cfg_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.pose_cfg_choice1)
        l_opt.addWidget(self.pose_cfg_choice2)
        self.layout_specify_edit.addLayout(l_opt)

    def _display(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Display iterations")
        self.display_iters_spin = QSpinBox()
        self.display_iters_spin.setValue(100)
        self.display_iters_spin.setMinimum(1)
        #self.display_iters_spin.setMaximum(int(self.max_iters))
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
        self.save_iters_spin.setValue(10000)
        self.save_iters_spin.setMinimum(1)
        #self.save_iters_spin.setMaximum(int(self.max_iters))
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
        self.max_iters_spin.setValue(50000)
        self.max_iters_spin.setMinimum(1)
        #self.max_iters_spin.setMaximum(int(self.max_iters))
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
        self.snapshots_spin = QSpinBox()
        self.snapshots_spin.setValue(5)
        self.snapshots_spin.setMinimum(1)
        self.snapshots_spin.setMaximum(100)
        self.snapshots_spin.setMinimumWidth(300)
        self.snapshots_spin.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.snapshots_spin)
        self.layout_display.addLayout(l_opt)


    def update_pose_cfg_choice(self, rb):
        if rb.text() == "Yes":
            self.pose_cfg_choice = True
        else:
            self.pose_cfg_choice = False




