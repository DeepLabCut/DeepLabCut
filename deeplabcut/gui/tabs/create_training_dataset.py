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

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import deeplabcut
import deeplabcut.compat as compat
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset import get_existing_shuffle_indices
from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.utils.auxiliaryfunctions import (
    get_data_and_metadata_filenames,
    get_training_set_folder,
)


class CreateTrainingDataset(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(CreateTrainingDataset, self).__init__(root, parent, h1_description)

        self.model_comparison = False

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.ok_button = QtWidgets.QPushButton("Create Training Dataset")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.create_training_dataset)

        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        if self.root.is_multianimal:
            func = deeplabcut.create_multianimaltraining_dataset
        else:
            func = deeplabcut.create_training_dataset
        label = QtWidgets.QLabel(func.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        # Shuffle
        shuffle_label = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Augmentation method
        augmentation_label = QtWidgets.QLabel("Augmentation method")
        methods = compat.get_available_aug_methods(self.root.project_engine)
        self.aug_choice = QtWidgets.QComboBox()
        self.aug_choice.addItems(methods)
        self.aug_choice.setCurrentText(methods[0])
        self.aug_choice.currentTextChanged.connect(self.log_augmentation_choice)

        # Neural Network
        nnet_label = QtWidgets.QLabel("Network architecture")
        self.net_choice = QtWidgets.QComboBox()

        if self.root.project_engine == Engine.TF:
            nets = DLCParams.NNETS.copy()
            if not self.root.is_multianimal:
                nets.remove("dlcrnet_ms5")
        else:
            # FIXME: Circular imports make it impossible to import this at the top
            from deeplabcut.pose_estimation_pytorch import available_models
            nets = available_models()

        self.net_choice.addItems(nets)
        default_net_type = self.root.cfg.get("default_net_type", "resnet_50")
        if default_net_type in nets:
            self.net_choice.setCurrentIndex(nets.index(default_net_type))
        self.net_choice.currentTextChanged.connect(self.log_net_choice)

        self.overwrite = QtWidgets.QCheckBox("Overwrite")
        self.overwrite.setChecked(False)
        self.overwrite.setToolTip(
            "When checked, creating a new shuffle with an index that already exists "
            "will overwrite the existing index. Be careful with this option as you "
            "might lose data."
        )
        self.overwrite.stateChanged.connect(
            lambda s: self.root.logger.info(f"Overwrite: {s}")
        )

        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(nnet_label, 0, 2)
        layout.addWidget(self.net_choice, 0, 3)
        layout.addWidget(augmentation_label, 0, 4)
        layout.addWidget(self.aug_choice, 0, 5)
        layout.addWidget(self.overwrite, 1, 0)

    def log_net_choice(self, net):
        self.root.logger.info(f"Network architecture set to {net.upper()}")

    def log_augmentation_choice(self, augmentation):
        self.root.logger.info(f"Image augmentation set to {augmentation.upper()}")

    def create_training_dataset(self):
        shuffle = self.shuffle.value()
        cfg = self.root.cfg
        existing_indices = get_existing_shuffle_indices(
            cfg=cfg,
            train_fraction=cfg["TrainingFraction"][self.root.trainingset_index]
        )

        overwrite = self.overwrite.isChecked()
        if shuffle in existing_indices:
            if overwrite:
                if not self._confirm_overwrite(shuffle, existing_indices):
                    return
            else:
                msg = _create_message_box(
                    f"The training dataset could not be created.",
                    (
                        f"Shuffle {shuffle} already exists - you can create a new "
                        "training dataset with an unused shuffle index (existing "
                        f"shuffles are {existing_indices}) or you can overwrite the "
                        f"shuffle by ticking the 'Overwrite' checkbox"
                    ),
                )
                msg.exec_()
                self.root.writer.write("Training dataset creation failed.")
                return

        if self.model_comparison:
            raise NotImplementedError
            # TODO: finish model_comparison
            deeplabcut.create_training_model_comparison(
                config_file,
                num_shuffles=shuffle,
                net_types=self.net_type,
                augmenter_types=self.aug_type,
            )
        else:
            if self.root.is_multianimal:
                deeplabcut.create_multianimaltraining_dataset(
                    self.root.config,
                    shuffle,
                    Shuffles=[self.shuffle.value()],
                    net_type=self.net_choice.currentText(),
                    userfeedback=not overwrite,
                )
            else:
                deeplabcut.create_training_dataset(
                    self.root.config,
                    shuffle,
                    Shuffles=[self.shuffle.value()],
                    net_type=self.net_choice.currentText(),
                    augmenter_type=self.aug_choice.currentText(),
                    userfeedback=not overwrite,
                )
            # Check that training data files were indeed created.
            trainingsetfolder = get_training_set_folder(self.root.cfg)
            filenames = list(
                get_data_and_metadata_filenames(
                    trainingsetfolder,
                    self.root.cfg["TrainingFraction"][0],
                    self.shuffle.value(),
                    self.root.cfg,
                )
            )
            if self.root.is_multianimal:
                filenames[0] = filenames[0].replace("mat", "pickle")
            if all(
                os.path.exists(os.path.join(self.root.project_folder, file))
                for file in filenames
            ):
                msg = _create_message_box(
                    "The training dataset is successfully created.",
                    "Use the function 'train_network' to start training. Happy training!",
                )
                msg.exec_()
                self.root.writer.write("Training dataset successfully created.")
            else:
                msg = _create_message_box(
                    "The training dataset could not be created.",
                    "Make sure there are annotated data under labeled-data.",
                )
                msg.exec_()
                self.root.writer.write("Training dataset creation failed.")

    def _confirm_overwrite(self, shuffle: int, existing_indices: list[int]) -> bool:
        """
        Asks the user to confirm that they want to overwrite a shuffle.

        Args:
            shuffle: the shuffle the user wants to overwrite
            existing_indices: the indices of existing shuffles

        Returns:
            whether the user confirmed overwriting the shuffle
        """
        try:
            engine = get_shuffle_engine(
                self.root.cfg, self.root.trainingset_index, shuffle
            )
            engine_str = f" (with engine '{engine.aliases[0]}')"
        except ValueError:
            engine_str = ""

        conf = _create_confirmation_box(
            title=f"Are you sure you want to overwrite shuffle {shuffle}?",
            description=(
                f"As shuffle {shuffle} already exists{engine_str}, "
                f"the training-dataset files would be overwritten."
            )
        )
        result = conf.exec()
        if result != QtWidgets.QMessageBox.Yes:
            msg = _create_message_box(
                text="The training dataset was not be created.",
                info_text=(
                    "You can create a shuffle with another index. Existing indices "
                    f"are {existing_indices}"
                ),
            )
            msg.exec_()
            self.root.writer.write("Training dataset creation interrupted.")
            return False

        return True


def _create_message_box(text, info_text):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info_text)

    msg.setWindowTitle("Info")
    msg.setMinimumWidth(900)
    logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg


def _create_confirmation_box(title, description):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(title)
    msg.setInformativeText(description)

    msg.setWindowTitle("Confirmation")
    msg.setMinimumWidth(900)
    logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    return msg
