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
import webbrowser
from functools import partial

from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS
from PySide6 import QtWidgets
from PySide6.QtCore import QRegularExpression, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap, QRegularExpressionValidator

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.gui import BASE_DIR
from deeplabcut.gui.components import (
    _create_grid_layout,
    _create_label_widget,
    DefaultTab,
    VideoSelectionWidget,
)
from deeplabcut.gui.utils import move_to_separate_thread
from deeplabcut.gui.widgets import ClickableLabel
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS


class RegExpValidator(QRegularExpressionValidator):
    validationChanged = Signal(QRegularExpressionValidator.State)

    def validate(self, input_, pos):
        state, input_, pos = super().validate(input_, pos)
        self.validationChanged.emit(state)
        return state, input_, pos


class ModelZoo(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super().__init__(root, parent, h1_description)
        self._val_pattern = QRegularExpression(r"(\d{3,5},\s*)+\d{3,5}")
        self._set_page()
        self.root.engine_change.connect(self._on_engine_change)
        self.root.engine_change.connect(self._update_available_models)
        self._destfolder = None

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self._build_common_attributes()
        self._build_tf_attributes()
        self._build_torch_attributes()

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.run_video_adaptation)
        self.main_layout.addWidget(self.run_button, alignment=Qt.AlignRight)

        self.home_button = QtWidgets.QPushButton("Return to Welcome page")
        self.home_button.clicked.connect(self.root._generate_welcome_page)
        self.main_layout.addWidget(self.home_button, alignment=Qt.AlignLeft)
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

        self.go_to_button = QtWidgets.QPushButton("Read Documentation")
        # go to url https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html#about-the-superanimal-models when button is clicked
        self.go_to_button.clicked.connect(
            lambda: webbrowser.open(
                "https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html#about-the-superanimal-models"
            )
        )
        self.main_layout.addWidget(self.go_to_button, alignment=Qt.AlignLeft)
        self._on_engine_change(self.root.engine)

    def _build_common_attributes(self) -> None:
        settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))
        section_title = _create_label_widget(
            "Supermodel Settings", "font:bold", (0, 50, 0, 0)
        )

        model_combo_text = QtWidgets.QLabel("Supermodel name")
        model_combo_text.setMinimumWidth(300)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(250)

        loc_label = ClickableLabel("Folder to store results:", parent=self)
        loc_label.signal.connect(self.select_folder)
        self.loc_line = QtWidgets.QLineEdit(self.root.project_folder, self)
        self.loc_line.setReadOnly(True)
        action = self.loc_line.addAction(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", "open2.png")),
            QtWidgets.QLineEdit.TrailingPosition,
        )
        action.triggered.connect(self.select_folder)

        settings_layout.addWidget(section_title, 0, 0)
        settings_layout.addWidget(model_combo_text, 1, 0)
        settings_layout.addWidget(self.model_combo, 1, 1)
        settings_layout.addWidget(loc_label, 2, 0)
        settings_layout.addWidget(self.loc_line, 2, 1)

        self.settings_widget = QtWidgets.QWidget()
        self.settings_widget.setLayout(settings_layout)
        self.main_layout.addWidget(self.settings_widget)

    def _build_tf_attributes(self) -> None:
        model_settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        scales_label = QtWidgets.QLabel("Scale list")
        scales_label.setMinimumWidth(300)
        self.scales_line = QtWidgets.QLineEdit("", parent=self)
        self.scales_line.setPlaceholderText(
            "Optionally input a list of integer sizes separated by commas..."
        )
        validator = RegExpValidator(self._val_pattern, self)
        validator.validationChanged.connect(self._handle_validation_change)
        self.scales_line.setValidator(validator)

        tooltip_label = QtWidgets.QLabel()
        tooltip_label.setPixmap(
            QPixmap(
                os.path.join(BASE_DIR, "assets", "icons", "help2.png")
            ).scaledToWidth(30)
        )
        tooltip_label.setToolTip(
            "Approximate animal sizes in pixels, for spatial pyramid search. If left "
            "blank, defaults to video height +/- 50 pixels"
        )

        self.adapt_checkbox = QtWidgets.QCheckBox("Use video adaptation")
        self.adapt_checkbox.setChecked(True)

        pseudo_threshold_label = QtWidgets.QLabel("Pseudo-label confidence threshold")
        self.pseudo_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.01,
            maximum=1.0,
            singleStep=0.05,
            value=0.1,
            wrapping=True,
        )
        self.pseudo_threshold_spinbox.setMaximumWidth(300)

        adapt_iter_label = QtWidgets.QLabel("Number of adaptation iterations")
        adapt_iter_label.setMinimumWidth(300)
        self.adapt_iter_spinbox = QtWidgets.QSpinBox()
        self.adapt_iter_spinbox.setRange(100, 10000)
        self.adapt_iter_spinbox.setValue(1000)
        self.adapt_iter_spinbox.setSingleStep(100)
        self.adapt_iter_spinbox.setGroupSeparatorShown(True)
        self.adapt_iter_spinbox.setMaximumWidth(300)

        model_settings_layout.addWidget(scales_label, 1, 0)
        model_settings_layout.addWidget(self.scales_line, 1, 1)
        model_settings_layout.addWidget(tooltip_label, 1, 2)
        model_settings_layout.addWidget(self.adapt_checkbox, 2, 0)
        model_settings_layout.addWidget(pseudo_threshold_label, 3, 0)
        model_settings_layout.addWidget(self.pseudo_threshold_spinbox, 3, 1)
        model_settings_layout.addWidget(adapt_iter_label, 4, 0)
        model_settings_layout.addWidget(self.adapt_iter_spinbox, 4, 1)
        self.tf_widget = QtWidgets.QWidget()
        self.tf_widget.setLayout(model_settings_layout)
        self.tf_widget.hide()
        self.main_layout.addWidget(self.tf_widget)

    def _build_torch_attributes(self) -> None:
        torch_settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        self.torch_adapt_checkbox = QtWidgets.QCheckBox("Use video adaptation")
        self.torch_adapt_checkbox.setChecked(True)

        pseudo_threshold_label = QtWidgets.QLabel("Pseudo-label confidence threshold")
        pseudo_threshold_label.setMinimumWidth(300)
        self.torch_pseudo_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.01,
            maximum=1.0,
            singleStep=0.05,
            value=0.1,
            wrapping=True,
        )
        self.torch_pseudo_threshold_spinbox.setMaximumWidth(300)

        adapt_epoch_label = QtWidgets.QLabel("Number of adaptation epochs")
        adapt_epoch_label.setMinimumWidth(300)
        self.torch_adapt_epoch_spinbox = QtWidgets.QSpinBox()
        self.torch_adapt_epoch_spinbox.setRange(1, 50)
        self.torch_adapt_epoch_spinbox.setValue(4)
        self.torch_adapt_epoch_spinbox.setMaximumWidth(300)

        adapt_det_epoch_label = QtWidgets.QLabel("Number of detector adaptation epochs")
        adapt_det_epoch_label.setMinimumWidth(300)
        self.torch_adapt_det_epoch_spinbox = QtWidgets.QSpinBox()
        self.torch_adapt_det_epoch_spinbox.setRange(1, 50)
        self.torch_adapt_det_epoch_spinbox.setValue(4)
        self.torch_adapt_det_epoch_spinbox.setMaximumWidth(300)

        torch_settings_layout.addWidget(self.torch_adapt_checkbox, 1, 0)
        torch_settings_layout.addWidget(pseudo_threshold_label, 2, 0)
        torch_settings_layout.addWidget(self.torch_pseudo_threshold_spinbox, 2, 1)
        torch_settings_layout.addWidget(adapt_epoch_label, 3, 0)
        torch_settings_layout.addWidget(self.torch_adapt_epoch_spinbox, 3, 1)
        torch_settings_layout.addWidget(adapt_det_epoch_label, 4, 0)
        torch_settings_layout.addWidget(self.torch_adapt_det_epoch_spinbox, 4, 1)
        self.torch_widget = QtWidgets.QWidget()
        self.torch_widget.setLayout(torch_settings_layout)
        self.torch_widget.hide()
        self.main_layout.addWidget(self.torch_widget)

    def select_folder(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Please select a folder", self.root.project_folder
        )
        if not dirname:
            return

        self._destfolder = dirname
        self.loc_line.setText(dirname)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(deeplabcut.video_inference_superanimal.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

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

        supermodel_name = self.model_combo.currentText()
        videotype = self.video_selection_widget.videotype_widget.currentText()
        kwargs = self._gather_kwargs()

        can_run_in_background = False
        if can_run_in_background:
            func = partial(
                deeplabcut.video_inference_superanimal,
                videos,
                supermodel_name,
                videotype=videotype,
                dest_folder=self._destfolder,
                **kwargs,
            )

            self.worker, self.thread = move_to_separate_thread(func)
            self.worker.finished.connect(self.signal_analysis_complete)
            self.thread.start()
            self.run_button.setEnabled(False)
            self.root._progress_bar.show()
        else:
            print(f"Calling video_inference_superanimal with kwargs={kwargs}")
            deeplabcut.video_inference_superanimal(
                videos,
                supermodel_name,
                videotype=videotype,
                dest_folder=self._destfolder,
                **kwargs,
            )
            self.signal_analysis_complete()

    def signal_analysis_complete(self):
        self.run_button.setEnabled(True)
        self.root._progress_bar.hide()
        msg = QtWidgets.QMessageBox(text="SuperAnimal video inference complete!")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()

    def _gather_kwargs(self) -> dict:
        kwargs = {}
        if self.root.engine == Engine.TF:
            scales = []
            scales_ = self.scales_line.text()
            if scales_:
                if (
                    self.scales_line.validator().validate(scales_, 0)[0]
                    == RegExpValidator.Acceptable
                ):
                    scales = list(map(int, scales_.split(",")))
            kwargs["scale_list"] = scales
            kwargs["video_adapt"] = self.adapt_checkbox.isChecked()
            kwargs["pseudo_threshold"] = self.pseudo_threshold_spinbox.value()
            kwargs["adapt_iterations"] = self.adapt_iter_spinbox.value()
        else:
            kwargs["video_adapt"] = self.torch_adapt_checkbox.isChecked()
            kwargs["pseudo_threshold"] = self.torch_pseudo_threshold_spinbox.value()
            kwargs["detector_epochs"] = self.torch_adapt_det_epoch_spinbox.value()
            kwargs["pose_epochs"] = self.torch_adapt_epoch_spinbox.value()

        return kwargs

    def _update_available_models(self, engine: Engine) -> None:
        while self.model_combo.count() > 0:
            self.model_combo.removeItem(0)

        supermodels = [
            model
            for model in MODELOPTIONS
            if (
                "superanimal" in model
                and model not in ("superanimal_topviewmouse", "superanimal_quadruped")
                and (
                    (engine == Engine.TF and "dlcrnet" in model)
                    or (engine == Engine.PYTORCH and "dlcrnet" not in model)
                )
            )
        ]
        self.model_combo.addItems(supermodels)

    @Slot(Engine)
    def _on_engine_change(self, engine: Engine) -> None:
        self._update_available_models(engine)
        if engine == Engine.PYTORCH:
            self.tf_widget.hide()
            self.torch_widget.show()
        else:
            self.torch_widget.hide()
            self.tf_widget.show()
