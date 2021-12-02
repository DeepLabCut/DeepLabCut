import os
import pydoc
import subprocess
import sys
import webbrowser

from PyQt5.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import deeplabcut
from deeplabcut.gui import LOGO_PATH
from deeplabcut.utils import auxiliaryfunctions

class Evaluate_network_page(QWidget):

    def __init__(self, parent, cfg):
        super(Evaluate_network_page, self).__init__(parent)

        self.method = "automatic"

        self.config = cfg
        self.plot_choice = False
        self.plot_scoremaps = False
        self.bodyparts = []

        self.inLayout = QtWidgets.QVBoxLayout(self)
        self.inLayout.setAlignment(Qt.AlignTop)
        self.inLayout.setSpacing(20)
        self.inLayout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(self.inLayout)

        self.set_page()

    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 6. Evaluate Network")
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

        label = QtWidgets.QLabel('Attributes')
        label.setContentsMargins(20, 20, 0, 10)
        self.layout_attributes.addWidget(label)

        self.layout_specify_plot = QtWidgets.QHBoxLayout()
        self.layout_specify_plot.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_specify_plot.setSpacing(20)
        self.layout_specify_plot.setContentsMargins(20, 0, 50, 20)

        self._specify_shuffle()
        self._specify_ts_index()
        self._plot_maps_choice()

        self.layout_predictions = QtWidgets.QHBoxLayout()
        self.layout_predictions.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_predictions.setSpacing(20)
        self.layout_predictions.setContentsMargins(20, 0, 50, 0)

        self._plot_predictions()
        self._compare_bp()

        self.layout_attributes.addLayout(self.layout_specify_plot)
        self.layout_attributes.addLayout(self.layout_predictions)

        self.ev_nw_button = QtWidgets.QPushButton('RUN: Evaluate Network')
        self.ev_nw_button.setMinimumWidth(200)
        self.ev_nw_button.setContentsMargins(0, 80, 40, 40)
        self.ev_nw_button.clicked.connect(self.evaluate_network)

        self.opt_button = QtWidgets.QPushButton('Optional: Plot 3 test maps')
        self.opt_button.setMinimumWidth(200)
        self.opt_button.setContentsMargins(0, 80, 40, 40)
        # self.opt_button.clicked.connect(self.plot_maps)

        self.layout_attributes.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)
        self.layout_attributes.addWidget(self.opt_button, alignment=Qt.AlignRight)

        self.inLayout.addLayout(self.layout_attributes)


    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):

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
        self.layout_specify_plot.addLayout(l_opt)
    def _specify_ts_index(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the trainingset index")
        self.trainingset = QSpinBox()
        self.trainingset.setValue(0)
        self.trainingset.setMinimumWidth(400)
        self.trainingset.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.trainingset)
        self.layout_specify_plot.addLayout(l_opt)
    def _plot_maps_choice(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Want to plot maps (ALL images): scoremaps, PAFs, locrefs?")
        self.btngroup_plot_maps_choice = QButtonGroup()

        self.plot_maps_choice1 = QtWidgets.QRadioButton('Yes')
        self.plot_maps_choice1.toggled.connect(lambda: self.update_map_choice(self.plot_maps_choice1))

        self.plot_maps_choice2 = QtWidgets.QRadioButton('No')
        self.plot_maps_choice2.setChecked(True)
        self.plot_maps_choice2.toggled.connect(lambda: self.update_map_choice(self.plot_maps_choice2))

        self.btngroup_plot_maps_choice.addButton(self.plot_maps_choice1)
        self.btngroup_plot_maps_choice.addButton(self.plot_maps_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.plot_maps_choice1)
        l_opt.addWidget(self.plot_maps_choice2)
        self.layout_specify_plot.addLayout(l_opt)

    def _plot_predictions(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Want to plot predictions (as in standard DLC projects)?")
        self.btngroup_plot_predictions_choice = QButtonGroup()

        self.plot_predictions_choice1 = QtWidgets.QRadioButton('Yes')
        self.plot_predictions_choice1.toggled.connect(lambda: self.update_plot_choice(self.plot_predictions_choice1))

        self.plot_predictions_choice2 = QtWidgets.QRadioButton('No')
        self.plot_predictions_choice2.setChecked(True)
        self.plot_predictions_choice2.toggled.connect(lambda: self.update_plot_choice(self.plot_predictions_choice2))

        self.btngroup_plot_predictions_choice.addButton(self.plot_predictions_choice1)
        self.btngroup_plot_predictions_choice.addButton(self.plot_predictions_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.plot_predictions_choice1)
        l_opt.addWidget(self.plot_predictions_choice2)
        self.layout_predictions.addLayout(l_opt)

    def _compare_bp(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(40, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Compare all bodyparts?")
        self.btngroup_compare_bp = QButtonGroup()

        self.compare_bp_choice1 = QtWidgets.QRadioButton('Yes')
        self.compare_bp_choice1.setChecked(True)
        self.compare_bp_choice1.toggled.connect(lambda: self.update_bp_choice(self.compare_bp_choice1))

        self.compare_bp_choice2 = QtWidgets.QRadioButton('No')
        self.compare_bp_choice2.toggled.connect(lambda: self.update_bp_choice(self.compare_bp_choice2))

        self.btngroup_compare_bp.addButton(self.compare_bp_choice1)
        self.btngroup_compare_bp.addButton(self.compare_bp_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.compare_bp_choice1)
        l_opt.addWidget(self.compare_bp_choice2)
        self.layout_predictions.addLayout(l_opt)

    def update_map_choice(self, rb):
        if rb.text() == "Yes":
            self.plot_scoremaps = True
            print('plot_scoremaps = T')
        else:
            self.plot_scoremaps = False
            print('plot_scoremaps = F')

    def update_plot_choice(self, rb):
        if rb.text() == "Yes":
            self.plot_choice = True
            print('plot_choice = T')
        else:
            self.plot_choice = False
            print('plot_choice = F')

    def update_bp_choice(self, rb):
        if rb.text() == "Yes":
            self.plot_bp_choice = True
            print('plot_bp_choice = T')

            self.bodyparts = "all"
            # self.bodyparts_to_compare.Hide()
            # self.SetSizer(self.sizer)
            # self.sizer.Fit(self)

        else:
            self.plot_bp_choice = False
            print('plot_bp_choice = F')
            # self.bodyparts_to_compare.Show()
            # self.getbp(event)
            # self.SetSizer(self.sizer)
            # self.sizer.Fit(self)

    def getbp(self):
        self.bodyparts = list()
        # self.bodyparts = list(self.bodyparts_to_compare.GetCheckedStrings())

    def evaluate_network(self):

        trainingsetindex = self.trainingset.value()

        Shuffles = [self.shuffles.value()]
        plotting = self.plot_choice

        if self.plot_scoremaps:
            for shuffle in Shuffles:
                deeplabcut.extract_save_all_maps(self.config, shuffle=shuffle)

        if len(self.bodyparts) == 0:
            self.bodyparts = "all"
        deeplabcut.evaluate_network(
            self.config,
            Shuffles=Shuffles,
            trainingsetindex=trainingsetindex,
            plotting=plotting,
            show_errors=True,
            comparisonbodyparts=self.bodyparts,
        )


