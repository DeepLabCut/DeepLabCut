import os

from PySide2.QtWidgets import QWidget
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon


from deeplabcut.generate_training_dataset import check_labels
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut import check_labels

from pathlib import Path

from components import DefaultTab, _create_horizontal_layout, _create_label_widget


class LabelFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(LabelFrames, self).__init__(root, parent, h1_description)

        self.set_page()
        

    def set_page(self):


        self.main_layout.addWidget(_create_label_widget("")) #dummy text

        self.label_frames_btn = QtWidgets.QPushButton("Label Frames")
        self.label_frames_btn.clicked.connect(self.label_frames)

        self.check_labels_btn = QtWidgets.QPushButton("Check Labels")
        self.check_labels_btn.clicked.connect(self.check_labels)
        self.check_labels_btn.setEnabled(True)

        if self.root.is_multianimal:
            self._add_color_by_option()
        
        self.main_layout.addWidget(self.label_frames_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.check_labels_btn, alignment=Qt.AlignRight)



    def _add_color_by_option(self):
        self.layout_multianimal_options = _create_horizontal_layout()

        self.color_by_widget = QtWidgets.QComboBox()
        self.color_by_widget.setMinimumWidth(150)
        self.color_by_widget.setMinimumHeight(30)
        options = ["individual", "bodypart"]
        self.color_by_widget.addItems(options)
        self.color_by_widget.currentTextChanged.connect(
            self.log_color_by_option
        )

        self.layout_multianimal_options.addWidget(QtWidgets.QLabel("Color labels by"))
        self.layout_multianimal_options.addWidget(self.color_by_widget)

        self.main_layout.addLayout(self.layout_multianimal_options)

    def log_color_by_option(self, choice):
        self.root.logger.info(f"Labeled images will by colored by {choice.upper()}")

    def check_labels(self):

        visualizeindividuals = False

        if self.root.is_multianimal:
            if self.color_by_widget.currentText() == "individual":
                visualizeindividuals = True
        
        check_labels(self.root.config, visualizeindividuals=visualizeindividuals)

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(
            "Labeled images have been created in project-folder/labeled-data/"
        )

        msg.setWindowTitle("Info")
        msg.setWindowIcon(QtWidgets.QMessageBox.Information)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def label_frames(self):
        if self.root.is_multianimal:
        # TODO:
            raise NotImplementedError

            import multiple_individuals_labeling_toolbox
            # multiple_individuals_labeling_toolbox.show(config, config3d, sourceCam)
        else:
            import labeling_toolbox
            labeling_frame = labeling_toolbox.MainFrame(
                self, 
                self.root.config, 
                imtypes=["*.png"], 
                config3d=None, 
                sourceCam=None,
            )
            labeling_frame.show()
        
