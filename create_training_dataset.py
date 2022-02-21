import os
from PyQt5.QtWidgets import QWidget, QComboBox, QSpinBox, QButtonGroup
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


class CreateTrainingDataset(QWidget):
    def __init__(self, parent, cfg):
        super(CreateTrainingDataset, self).__init__(parent)

        self.method = "automatic"
        self.userfeedback = False
        self.model_comparison = False
        self.config = cfg

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Raised)

        separatorLine.setLineWidth(0)
        separatorLine.setMidLineWidth(1)

        inLayout = QtWidgets.QVBoxLayout(self)
        inLayout.setAlignment(Qt.AlignTop)
        inLayout.setSpacing(20)
        inLayout.setContentsMargins(0, 20, 0, 20)
        self.setLayout(inLayout)

        l1_step1 = QtWidgets.QLabel("DeepLabCut - Step 4. Create training dataset")
        l1_step1.setContentsMargins(20, 0, 0, 10)

        inLayout.addWidget(l1_step1)
        inLayout.addWidget(separatorLine)

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

        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.browse_dir)

        layout_cfg.addWidget(cfg_text)
        layout_cfg.addWidget(self.cfg_line)
        layout_cfg.addWidget(browse_button)

        inLayout.addLayout(layout_cfg)

        self.layout_attributes = QtWidgets.QVBoxLayout()
        self.layout_attributes.setAlignment(Qt.AlignTop)
        self.layout_attributes.setSpacing(20)
        self.layout_attributes.setContentsMargins(0, 0, 40, 0)

        label = QtWidgets.QLabel("Optional Attributes")
        label.setContentsMargins(20, 20, 0, 10)
        self.layout_attributes.addWidget(label)

        self.layout_select_network_method = QtWidgets.QHBoxLayout()
        self.layout_select_network_method.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.layout_select_network_method.setSpacing(0)
        self.layout_select_network_method.setContentsMargins(20, 0, 50, 0)

        self._select_network_method()
        self._select_aug()

        self.layout_set_indx = QtWidgets.QHBoxLayout()
        self.layout_set_indx.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.layout_set_indx.setSpacing(0)
        self.layout_set_indx.setContentsMargins(20, 0, 50, 0)

        self._set_shuffle()
        self._specify_ts_ind()

        self.layout_feedback_compare = QtWidgets.QHBoxLayout()
        self.layout_feedback_compare.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout_feedback_compare.setSpacing(0)
        self.layout_feedback_compare.setContentsMargins(20, 0, 50, 0)

        self._userfeedback()
        self._compare_mdls()

        self.layout_attributes.addLayout(self.layout_select_network_method)
        self.layout_attributes.addLayout(self.layout_set_indx)
        self.layout_attributes.addLayout(self.layout_feedback_compare)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.setContentsMargins(0, 40, 40, 40)
        self.ok_button.clicked.connect(self.create_training_dataset)

        self.layout_attributes.addWidget(self.ok_button, alignment=Qt.AlignRight)

        inLayout.addLayout(self.layout_attributes)

    def update_cfg(self):
        text = self.proj_line.text()
        self.config = text

    def browse_dir(self):
        cwd = self.config
        config = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a configuration file", cwd, "Config files (*.yaml)"
        )

        if not config[0]:
            return
        self.config = config[0]
        self.cfg_line.setText(self.config)

    def _select_network_method(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Select the network")
        self.net_choice = QComboBox()
        self.net_choice.setMinimumWidth(550)
        self.net_choice.setMinimumHeight(30)
        options = [
            "dlcrnet_ms5",
            "resnet_50",
            "resnet_101",
            "resnet_152",
            "mobilenet_v2_1.0",
            "mobilenet_v2_0.75",
            "mobilenet_v2_0.5",
            "mobilenet_v2_0.35",
            "efficientnet-b0",
            "efficientnet-b3",
            "efficientnet-b6",
        ]
        self.net_choice.addItems(options)
        self.net_choice.setCurrentText("resnet_50")

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.net_choice)
        self.layout_select_network_method.addLayout(l_opt, 1)

    def _select_aug(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Select the augmentation method")
        self.aug_choice = QComboBox()
        self.aug_choice.setMinimumWidth(550)
        self.aug_choice.setMinimumHeight(30)
        options = ["default", "tensorpack", "imgaug"]
        self.aug_choice.addItems(options)
        self.aug_choice.setCurrentText("imgaug")

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.aug_choice)
        l_opt.addStretch()
        self.layout_select_network_method.addLayout(l_opt, 1)

    def _set_shuffle(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Set a specific shuffle indx (1 network only)")
        self.shuffle = QSpinBox()
        self.shuffle.setValue(1)
        self.shuffle.setMinimumWidth(550)
        self.shuffle.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.shuffle)
        self.layout_set_indx.addLayout(l_opt, 1)

    def _specify_ts_ind(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Specify the trainingset index")
        self.trainingindex = QSpinBox()
        self.trainingindex.setValue(0)
        self.trainingindex.setMinimumWidth(550)
        self.trainingindex.setMinimumHeight(30)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.trainingindex)
        self.layout_set_indx.addLayout(l_opt, 1)

    def _userfeedback(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel(
            "User feedback (to confirm overwrite train/test split)?"
        )
        self.btngroup_userfeedback = QButtonGroup()

        self.feedback_1 = QtWidgets.QRadioButton("Yes")
        self.feedback_1.toggled.connect(
            lambda: self.update_feedback_choice(self.feedback_1)
        )

        self.feedback_2 = QtWidgets.QRadioButton("No")
        self.feedback_2.setChecked(True)
        self.feedback_2.toggled.connect(
            lambda: self.update_feedback_choice(self.feedback_2)
        )

        self.btngroup_userfeedback.addButton(self.feedback_1)
        self.btngroup_userfeedback.addButton(self.feedback_2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.feedback_1)
        l_opt.addWidget(self.feedback_2)

        self.layout_feedback_compare.addLayout(l_opt, 1)

    def _compare_mdls(self):
        l_opt = QtWidgets.QVBoxLayout()
        l_opt.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        l_opt.setSpacing(20)
        l_opt.setContentsMargins(20, 0, 0, 0)

        opt_text = QtWidgets.QLabel("Want to compare models?")
        self.btngroup_comparison = QButtonGroup()

        self.model_comparison_choice1 = QtWidgets.QRadioButton("Yes")
        self.model_comparison_choice1.toggled.connect(
            lambda: self.update_model_comparison_choice(self.model_comparison_choice1)
        )

        self.model_comparison_choice2 = QtWidgets.QRadioButton("No")
        self.model_comparison_choice2.setChecked(True)
        self.model_comparison_choice2.toggled.connect(
            lambda: self.update_model_comparison_choice(self.model_comparison_choice2)
        )

        self.btngroup_comparison.addButton(self.model_comparison_choice1)
        self.btngroup_comparison.addButton(self.model_comparison_choice2)

        l_opt.addWidget(opt_text)
        l_opt.addWidget(self.model_comparison_choice1)
        l_opt.addWidget(self.model_comparison_choice2)

        self.layout_feedback_compare.addLayout(l_opt, 1)

    def update_feedback_choice(self, rb):
        if rb.text() == "Yes":
            self.userfeedback = True
        else:
            self.userfeedback = False
        # TODO: add functionality

    def update_model_comparison_choice(self, rb):
        if rb.text() == "Yes":
            self.model_comparison = True
        else:
            self.model_comparison = False
        # TODO: add functionality

    def create_training_dataset(self):
        num_shuffles = self.shuffle.value()
        config_file = auxiliaryfunctions.read_config(self.config)
        trainindex = self.trainingindex.value()

        userfeedback = self.userfeedback

        if config_file.get("multianimalproject", False):
            print("multianimalproject")
            # TODO: add multianimal part
            # deeplabcut.create_multianimaltraining_dataset(
            #     self.config,
            #     num_shuffles,
            #     Shuffles=[self.shuffle.value()],
            #     net_type=self.net_choice.currentText() ,
            # )
        else:
            if self.model_comparison == False:
                deeplabcut.create_training_dataset(
                    self.config,
                    num_shuffles,
                    Shuffles=[self.shuffle.value()],
                    userfeedback=userfeedback,
                    net_type=self.net_choice.currentText(),
                    augmenter_type=self.aug_choice.currentText(),
                )
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText("The training dataset is successfully created.")
                msg.setInformativeText(
                    "Use the function 'train_network' to start training. Happy training!"
                )

                msg.setWindowTitle("Info")
                msg.setMinimumWidth(900)
                self.logo_dir = (
                    os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
                )
                self.logo = self.logo_dir + "/assets/logo.png"
                msg.setWindowIcon(QIcon(self.logo))
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

            else:
                # comparison = True
                # TODO: finish model_comparison
                deeplabcut.create_training_model_comparison(
                    self.config,
                    trainindex=trainindex,
                    num_shuffles=num_shuffles,
                    userfeedback=userfeedback,
                    net_types=self.net_type,
                    augmenter_types=self.aug_type,
                )
