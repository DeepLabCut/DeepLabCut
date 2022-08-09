import napari
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from deeplabcut_gui.components import (
    DefaultTab, _create_horizontal_layout, _create_label_widget,
)


class LabelFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(LabelFrames, self).__init__(root, parent, h1_description)

        self._set_page()

    def _set_page(self):

        self.main_layout.addWidget(_create_label_widget(""))  # dummy text

        self.label_frames_btn = QtWidgets.QPushButton("Label Frames")
        self.label_frames_btn.clicked.connect(self.label_frames)

        if self.root.is_multianimal:
            self.layout_multianimal = _create_horizontal_layout()
            self._generate_layout_multianimal(self.layout_multianimal)
            self.main_layout.addLayout(self.layout_multianimal)

        self.main_layout.addWidget(self.label_frames_btn, alignment=Qt.AlignRight)

    def _generate_layout_multianimal(self, layout):

        self.color_by_widget = QtWidgets.QComboBox()
        self.color_by_widget.setMinimumWidth(150)
        options = ["individual", "bodypart"]
        self.color_by_widget.addItems(options)
        self.color_by_widget.currentTextChanged.connect(self.log_color_by_option)

        layout.addWidget(QtWidgets.QLabel("Color labels by"))
        layout.addWidget(self.color_by_widget)

    def log_color_by_option(self, choice):
        self.root.logger.info(f"Labeled images will by colored by {choice.upper()}")

    def label_frames(self):
        _ = napari.Viewer()
