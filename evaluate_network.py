import os
import subprocess
import sys
import webbrowser

from PySide2.QtWidgets import QWidget, QSpinBox, QButtonGroup
from PySide2 import QtWidgets
from PySide2.QtCore import Qt

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from components import _create_horizontal_layout, _create_label_widget


class EvaluateNetwork(QWidget):
    def __init__(self, parent, cfg):
        super(EvaluateNetwork, self).__init__(parent)

        self.method = "automatic"

        self.config = cfg
        self.plot_choice = False
        self.plot_scoremaps = False
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.bodyparts = []

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(0, 20, 10, 20)
        self.setLayout(self.main_layout)

        self.set_page()

    def set_page(self):
        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 6. Evaluate Network")
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

        self.layout_predictions = QtWidgets.QHBoxLayout()
        self.layout_predictions.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout_predictions.setSpacing(20)
        self.layout_predictions.setContentsMargins(20, 0, 50, 0)

        self._plot_predictions()
        self._compare_bp()

        self.layout_attributes.addLayout(self.layout_predictions)

        self.ev_nw_button = QtWidgets.QPushButton("Evaluate Network")
        self.ev_nw_button.setMinimumWidth(200)
        self.ev_nw_button.setContentsMargins(0, 80, 40, 40)
        self.ev_nw_button.clicked.connect(self.evaluate_network)

        self.opt_button = QtWidgets.QPushButton("Plot 3 test maps")
        self.opt_button.setMinimumWidth(200)
        self.opt_button.setContentsMargins(0, 80, 40, 40)
        self.opt_button.clicked.connect(self.plot_maps)

        self.layout_attributes.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)
        self.layout_attributes.addWidget(self.opt_button, alignment=Qt.AlignRight)

        self.main_layout.addLayout(self.layout_attributes)
        # TODO: finish multianimal part:
        # if config_file.get("multianimalproject", False):
        #     self.inf_cfg_text = wx.Button(self, label="Edit the inference_config.yaml")
        #     self.inf_cfg_text.Bind(wx.EVT_BUTTON, self.edit_inf_config)
        #     self.sizer.Add(
        #         self.inf_cfg_text,
        #         pos=(4, 2),
        #         span=(1, 1),
        #         flag=wx.BOTTOM | wx.RIGHT,
        #         border=10,
        #     )

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
        self.trainingset = QSpinBox()
        self.trainingset.setMaximum(100)
        self.trainingset.setValue(0)
        self.trainingset.setMinimumHeight(30)

        layout.addWidget(opt_text)
        layout.addWidget(self.trainingset)

        # self.plot_maps_checkbox = QtWidgets.QCheckBox("Plot maps (scoremaps, PAFs, locref)")
        # self.plot_maps_checkbox.setCheckState(Qt.Unchecked)
        # self.plot_maps_checkbox.stateChanged.connect(self.update_map_choice)
        # layout.addWidget(self.plot_maps_checkbox)



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
        text = self.proj_line.text()
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

    def plot_maps(self):
        shuffle = self.shuffle.value()
        deeplabcut.extract_save_all_maps(
            self.config, shuffle=shuffle, Indices=[0, 1, 5]
        )        

    def _plot_predictions(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel(
            "Want to plot predictions (as in standard DLC projects)?"
        )
        self.btngroup_plot_predictions_choice = QButtonGroup()

        self.plot_predictions_choice1 = QtWidgets.QRadioButton("Yes")
        self.plot_predictions_choice1.toggled.connect(
            lambda: self.update_plot_choice(self.plot_predictions_choice1)
        )

        self.plot_predictions_choice2 = QtWidgets.QRadioButton("No")
        self.plot_predictions_choice2.setChecked(True)
        self.plot_predictions_choice2.toggled.connect(
            lambda: self.update_plot_choice(self.plot_predictions_choice2)
        )

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

        # if self.cfg.get("multianimalproject", False):
        #     self.bodyparts = self.config["multianimalbodyparts"]
        # else:
        #     self.bodyparts = self.config["bodyparts"]

        opt_text = QtWidgets.QLabel("Compare all bodyparts?")
        self.btngroup_compare_bp = QButtonGroup()

        self.compare_bp_choice1 = QtWidgets.QRadioButton("Yes")
        self.compare_bp_choice1.setChecked(True)
        self.compare_bp_choice1.toggled.connect(
            lambda: self.update_bp_choice(self.compare_bp_choice1)
        )

        self.compare_bp_choice2 = QtWidgets.QRadioButton("No")
        self.compare_bp_choice2.toggled.connect(
            lambda: self.update_bp_choice(self.compare_bp_choice2)
        )

        self.btngroup_compare_bp.addButton(self.compare_bp_choice1)
        self.btngroup_compare_bp.addButton(self.compare_bp_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.compare_bp_choice1)
        l_opt.addWidget(self.compare_bp_choice2)
        self.layout_predictions.addLayout(l_opt)

    def update_map_choice(self, state):
        if state == Qt.Checked:
            self.plot_scoremaps = True
        else:
            self.plot_scoremaps = False

    def update_plot_choice(self, rb):
        if rb.text() == "Yes":
            self.plot_choice = True

        else:
            self.plot_choice = False

    def update_bp_choice(self, rb):
        # TODO: finish functionality
        if rb.text() == "Yes":
            self.plot_bp_choice = True
            self.bodyparts = "all"
            # self.bodyparts_to_compare.Hide()
            # self.SetSizer(self.sizer)
            # self.sizer.Fit(self)

        else:
            self.plot_bp_choice = False
            # self.bodyparts_to_compare.Show()
            # self.getbp(event)
            # self.SetSizer(self.sizer)
            # self.sizer.Fit(self)

    def getbp(self):
        self.bodyparts = list()
        # self.bodyparts = list(self.bodyparts_to_compare.GetCheckedStrings())

    def edit_inf_config(self, event):
        # Read the infer config file
        cfg = auxiliaryfunctions.read_config(self.config)
        trainingsetindex = self.trainingset.value()
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
        self.inf_cfg_path = os.path.join(
            cfg["project_path"],
            auxiliaryfunctions.GetModelFolder(
                trainFraction, self.shuffle.value(), cfg
            ),
            "test",
            "inference_cfg.yaml",
        )
        # let the user open the file with default text editor. Also make it mac compatible
        if sys.platform == "darwin":
            self.file_open_bool = subprocess.call(["open", self.inf_cfg_path])
            self.file_open_bool = True
        else:
            self.file_open_bool = webbrowser.open(self.inf_cfg_path)
        if self.file_open_bool:
            self.inf_cfg = auxiliaryfunctions.read_config(self.inf_cfg_path)
        else:
            raise FileNotFoundError("File not found!")

    def evaluate_network(self):

        trainingsetindex = self.trainingset.value()

        Shuffles = [self.shuffle.value()]
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
