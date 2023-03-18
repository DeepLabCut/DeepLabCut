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
import deeplabcut
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QRegularExpressionValidator
from deeplabcut.gui.components import (
    DefaultTab,
    VideoSelectionWidget,
    _create_label_widget,
    _create_grid_layout,
)
from deeplabcut.modelzoo.utils import parse_available_supermodels


class RegExpValidator(QRegularExpressionValidator):
    validationChanged = Signal(QRegularExpressionValidator.State)

    def validate(self, input_, pos):
        state, input_, pos = super().validate(input_, pos)
        self.validationChanged.emit(state)
        return state, input_, pos


class ModelZoo(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super().__init__(root, parent, h1_description)
        self._val_pattern = "(\d{3,5},\s*)+\d{3,5}"
        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        model_settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        section_title = _create_label_widget(
            "Supermodel Settings", "font:bold", (0, 50, 0, 0)
        )

        model_combo_text = QtWidgets.QLabel("Supermodel name")
        self.model_combo = QtWidgets.QComboBox()
        supermodels = parse_available_supermodels()
        self.model_combo.addItems(supermodels.keys())

        scales_label = QtWidgets.QLabel("Scale list")
        self.scales_line = QtWidgets.QLineEdit("", parent=self)
        self.scales_line.setPlaceholderText(
            "Optionally input a list of integer sizes separated by commas..."
        )
        validator = RegExpValidator(self._val_pattern, self)
        validator.validationChanged.connect(self._handle_validation_change)
        self.scales_line.setValidator(validator)

        model_settings_layout.addWidget(section_title, 0, 0)
        model_settings_layout.addWidget(model_combo_text, 1, 0)
        model_settings_layout.addWidget(self.model_combo, 1, 1)
        model_settings_layout.addWidget(scales_label, 2, 0)
        model_settings_layout.addWidget(self.scales_line, 2, 1)
        self.main_layout.addLayout(model_settings_layout)

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.run_video_adaptation)
        self.main_layout.addWidget(self.run_button, alignment=Qt.AlignRight)

    def _handle_validation_change(self, state):
        if state == RegExpValidator.Invalid:
            color = "red"
        elif state == RegExpValidator.Intermediate:
            color = "gold"
        elif state == RegExpValidator.Acceptable:
            color = "lime"
        self.scales_line.setStyleSheet(f"border: 1px solid {color}")
        QTimer.singleShot(500, lambda: self.scales_line.setStyleSheet(""))

    def run_video_adaptation(self):
        videos = list(self.files)
        if not videos:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("You must select a video file")
            msg.setWindowTitle("Error")
            msg.setMinimumWidth(400)
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        scales = []
        scales_ = self.scales_line.text()
        if scales_:
            if (
                self.scales_line.validator().validate(scales_, 0)[0]
                == RegExpValidator.Acceptable
            ):
                scales = list(map(int, scales_.split(",")))
        supermodel_name = self.model_combo.currentText()
        videotype = self.video_selection_widget.videotype_widget.currentText()

        deeplabcut.video_inference_superanimal(
            videos,
            supermodel_name,
            videotype=videotype,
            video_adapt=True,
            scale_list=scales,
        )
