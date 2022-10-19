import os

from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon

from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)

import deeplabcut
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

    def _generate_layout_attributes(self, layout):
        # Shuffle
        shuffle_label = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Augmentation method
        augmentation_label = QtWidgets.QLabel("Augmentation method")
        self.aug_choice = QtWidgets.QComboBox()
        self.aug_choice.addItems(DLCParams.IMAGE_AUGMENTERS)
        self.aug_choice.setCurrentText("imgaug")
        self.aug_choice.currentTextChanged.connect(self.log_augmentation_choice)

        # Neural Network
        nnet_label = QtWidgets.QLabel("Network architecture")
        self.net_choice = QtWidgets.QComboBox()
        nets = DLCParams.NNETS.copy()
        if not self.root.is_multianimal:
            nets.remove('dlcrnet_ms5')
        self.net_choice.addItems(nets)
        self.net_choice.setCurrentText("resnet_50")
        self.net_choice.currentTextChanged.connect(self.log_net_choice)

        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(nnet_label, 0, 2)
        layout.addWidget(self.net_choice, 0, 3)
        layout.addWidget(augmentation_label, 0, 4)
        layout.addWidget(self.aug_choice, 0, 5)

    def log_net_choice(self, net):
        self.root.logger.info(f"Network architecture set to {net.upper()}")

    def log_augmentation_choice(self, augmentation):
        self.root.logger.info(f"Image augmentation set to {augmentation.upper()}")

    def create_training_dataset(self):
        shuffle = self.shuffle.value()

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
                )
            else:
                deeplabcut.create_training_dataset(
                    self.root.config,
                    shuffle,
                    Shuffles=[self.shuffle.value()],
                    net_type=self.net_choice.currentText(),
                    augmenter_type=self.aug_choice.currentText(),
                )
            # Check that training data files were indeed created.
            trainingsetfolder = get_training_set_folder(self.root.cfg)
            filenames = list(get_data_and_metadata_filenames(
                trainingsetfolder,
                self.root.cfg["TrainingFraction"][0],
                self.shuffle.value(),
                self.root.cfg,
            ))
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


def _create_message_box(text, info_text):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info_text)

    msg.setWindowTitle("Info")
    msg.setMinimumWidth(900)
    logo_dir = (
            os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    )
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg
