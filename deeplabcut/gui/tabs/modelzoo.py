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
from pathlib import Path

import dlclibrary
from PySide6 import QtWidgets
from PySide6.QtCore import QRegularExpression, Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QPixmap, QRegularExpressionValidator
import cv2
import torch

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.gui import BASE_DIR
from deeplabcut.gui.components import (
    _create_grid_layout,
    _create_label_widget,
    DefaultTab,
    MediaSelectionWidget,
    set_layout_contents_visible,
    set_combo_items,
)
from deeplabcut.gui.utils import move_to_separate_thread
from deeplabcut.gui.widgets import ClickableLabel


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
        self._update_pose_models(self.model_combo.currentText())
        self._update_detectors(self.model_combo.currentText())
        self._destfolder = None
        self.worker = None
        self.thread = None
        
    @property
    def files(self):
        return self.media_selection_widget.files

    def _set_page(self):
        # Create Run button first so it exists for any method that references it
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_button.setFixedWidth(120)
        self.run_button.clicked.connect(self.run_video_inference_superanimal)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addStretch()

        self.main_layout.addWidget(_create_label_widget("Media Selection", "font:bold"))
        self.media_selection_widget = MediaSelectionWidget(self.root, self, hide_videotype=True)
        self.main_layout.addWidget(self.media_selection_widget)

        # Remove/hide image selection widgets (not needed for modelzoo)
        self.media_selection_widget.media_type_widget.hide()
        self.media_selection_widget.media_type_widget.setCurrentText("Videos")

        self._build_common_attributes()
        self._build_tf_attributes()
        self._build_torch_attributes()
        
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
        
        # Add the Run button layout
        self.main_layout.addLayout(button_layout)
        
        self._on_engine_change(self.root.engine)

    def _add_supermodel_section(self, layout: QtWidgets.QGridLayout) -> None:
        # --- Supermodel selection ---
        section_title = QtWidgets.QLabel("Supermodel settings")
        section_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        model_combo_text = QtWidgets.QLabel("Supermodel")
        model_combo_text.setMinimumWidth(150)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(250)
        layout.addWidget(section_title, 0, 0, 1, 6)
        layout.addWidget(model_combo_text, 1, 0)
        layout.addWidget(self.model_combo, 1, 1)

    def _add_pose_model_settings_row(self, layout: QtWidgets.QGridLayout):
        # --- Pose Model Type and Pose Confidence Threshold on the same line (now row 2) ---
        pose_model_row = QtWidgets.QHBoxLayout()
        pose_model_label = QtWidgets.QLabel("Pose Model Type")
        pose_model_label.setMinimumWidth(150)
        self.net_type_selector = QtWidgets.QComboBox()
        self.net_type_selector.setMinimumWidth(180)
        pose_conf_label = QtWidgets.QLabel("Pose confidence threshold")
        pose_conf_label.setMinimumWidth(170)
        self.pose_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.0,
            maximum=1.0,
            singleStep=0.01,
            value=0.4,
            wrapping=True,
        )
        self.pose_threshold_spinbox.setMaximumWidth(100)
        batch_size_combo_label = QtWidgets.QLabel("Pose model batch size")
        self.batch_size_combo = QtWidgets.QComboBox()
        self.batch_size_combo.setMinimumWidth(100)
        self.batch_size_combo.addItems([str(2 ** i) for i in range(6)])
        self.batch_size_combo.setCurrentIndex(0)
        pose_model_row.addWidget(pose_model_label)
        pose_model_row.addWidget(self.net_type_selector)
        pose_model_row.addSpacing(20)
        pose_model_row.addWidget(pose_conf_label)
        pose_model_row.addWidget(self.pose_threshold_spinbox)
        pose_model_row.addSpacing(20)
        pose_model_row.addWidget(batch_size_combo_label)
        pose_model_row.addWidget(self.batch_size_combo)
        pose_model_row.addStretch()
        layout.addLayout(pose_model_row, 2, 0, 1, 6)

    def _add_detector_settings_row(self, layout: QtWidgets.QGridLayout):
        # --- Detector Type and Detector Confidence Threshold on the same line (now row 3) ---
        detector_label = QtWidgets.QLabel("Detector Type")
        detector_label.setMinimumWidth(150)
        self.detector_type_selector = QtWidgets.QComboBox()
        self.detector_type_selector.setMinimumWidth(180)
        detector_conf_label = QtWidgets.QLabel("Detector confidence threshold")
        detector_conf_label.setMinimumWidth(170)
        self.detector_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.0,
            maximum=1.0,
            singleStep=0.01,
            value=0.1,
            wrapping=True,
        )
        self.detector_threshold_spinbox.setMaximumWidth(100)
        max_individuals_label = QtWidgets.QLabel("Maximum number of individuals")
        max_individuals_label.setMinimumWidth(180)
        self.max_individuals_spinbox = QtWidgets.QSpinBox()
        self.max_individuals_spinbox.setRange(1, 100)
        self.max_individuals_spinbox.setValue(1)
        self.max_individuals_spinbox.setMaximumWidth(100)
        detector_batch_size_combo_label = QtWidgets.QLabel("Detector batch size")
        self.detector_batch_size_combo = QtWidgets.QComboBox()
        self.detector_batch_size_combo.setMinimumWidth(100)
        self.detector_batch_size_combo.addItems([str(2 ** i) for i in range(6)])
        self.detector_batch_size_combo.setCurrentIndex(0)
        self.detector_row = QtWidgets.QHBoxLayout()
        self.detector_row.addWidget(detector_label)
        self.detector_row.addWidget(self.detector_type_selector)
        self.detector_row.addSpacing(20)
        self.detector_row.addWidget(detector_conf_label)
        self.detector_row.addWidget(self.detector_threshold_spinbox)
        self.detector_row.addSpacing(20)
        self.detector_row.addWidget(max_individuals_label)
        self.detector_row.addWidget(self.max_individuals_spinbox)
        self.detector_row.addSpacing(20)
        self.detector_row.addWidget(detector_batch_size_combo_label)
        self.detector_row.addWidget(self.detector_batch_size_combo)
        self.detector_row.addStretch()
        layout.addLayout(self.detector_row, 3, 0, 1, 6)

    def _add_output_settings_section(self, layout: QtWidgets.QGridLayout):
        loc_label = ClickableLabel("Folder to store results:", parent=self)
        loc_label.signal.connect(self.select_folder)
        self.loc_line = QtWidgets.QLineEdit(
            "<Select a folder - Default: store in same folder as media>",
            self,
        )
        self.loc_line.setReadOnly(True)
        action = self.loc_line.addAction(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", "open2.png")),
            QtWidgets.QLineEdit.TrailingPosition,
        )
        action.triggered.connect(self.select_folder)

        self.create_labeled_video_checkbox = QtWidgets.QCheckBox("Create labeled video")
        self.create_labeled_video_checkbox.setChecked(True)

        layout.addWidget(loc_label, 4, 0)
        layout.addWidget(self.loc_line, 4, 1)
        layout.addWidget(self.create_labeled_video_checkbox, 5, 0)

    def _build_common_attributes(self) -> None:
        settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        self._add_supermodel_section(settings_layout)
        self._add_pose_model_settings_row(settings_layout)
        self._add_detector_settings_row(settings_layout)
        self._add_output_settings_section(settings_layout)

        self.settings_widget = QtWidgets.QWidget()
        self.settings_widget.setLayout(settings_layout)
        self.main_layout.addWidget(self.settings_widget)

        self.model_combo.currentTextChanged.connect(self._update_pose_models)
        self.model_combo.currentTextChanged.connect(self._update_detectors)
        self.model_combo.currentTextChanged.connect(self._update_adaptation_visibility)

    def _add_tf_scales_row(self, layout: QtWidgets.QGridLayout):
        scales_label = QtWidgets.QLabel("Scale list")
        scales_label.setMinimumWidth(300)
        self.scales_line = QtWidgets.QLineEdit("", parent=self)
        self.scales_line.setMinimumWidth(500)
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
        scales_row = QtWidgets.QHBoxLayout()
        scales_row.addWidget(scales_label)
        scales_row.addWidget(self.scales_line)
        scales_row.addWidget(tooltip_label)
        layout.addLayout(scales_row, 1, 0, 1, 2)

    def _add_tf_use_adaptation_row(self, layout: QtWidgets.QGridLayout):
        # --- Adaptation Checkbox with Help Button (TF section) ---
        self.adapt_checkbox = QtWidgets.QCheckBox("Use video adaptation")
        self.adapt_checkbox.setChecked(True)
        self.adapt_checkbox.setStyleSheet("font-weight: bold; font-size: 16px; padding: 6px 12px;")
        # Add help button
        adapt_help_btn = QtWidgets.QToolButton()
        adapt_help_btn.setIcon(QIcon(os.path.join(BASE_DIR, "assets", "icons", "help2.png")))
        adapt_help_btn.setIconSize(QSize(24, 24))
        adapt_help_btn.setToolTip("What is video adaptation?")
        def show_adapt_help():
            QtWidgets.QMessageBox.information(
                self,
                "Video Adaptation",
                "This will adapt the model on the fly to your video data in a self-supervised way."
            )
        adapt_help_btn.clicked.connect(show_adapt_help)
        use_adaptation_row = QtWidgets.QHBoxLayout()
        use_adaptation_row.addWidget(self.adapt_checkbox)
        use_adaptation_row.addWidget(adapt_help_btn)
        use_adaptation_row.addStretch()
        layout.addLayout(use_adaptation_row, 2, 0, 1, 2)

    def _add_tf_adaptation_settings_row(self, layout: QtWidgets.QGridLayout):
        pseudo_threshold_label = QtWidgets.QLabel("Pseudo-label confidence threshold")
        self.pseudo_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.01,
            maximum=1.0,
            singleStep=0.05,
            value=0.1,
            wrapping=True,
        )
        self.pseudo_threshold_spinbox.setMaximumWidth(100)
        adapt_iter_label = QtWidgets.QLabel("Number of adaptation iterations")
        adapt_iter_label.setMinimumWidth(300)
        self.adapt_iter_spinbox = QtWidgets.QSpinBox()
        self.adapt_iter_spinbox.setRange(100, 10000)
        self.adapt_iter_spinbox.setValue(1000)
        self.adapt_iter_spinbox.setSingleStep(100)
        self.adapt_iter_spinbox.setGroupSeparatorShown(True)
        self.adapt_iter_spinbox.setMaximumWidth(300)
        self.tf_adaptation_settings_row = QtWidgets.QHBoxLayout()
        self.tf_adaptation_settings_row.addWidget(pseudo_threshold_label)
        self.tf_adaptation_settings_row.addWidget(self.pseudo_threshold_spinbox)
        self.tf_adaptation_settings_row.addSpacing(20)
        self.tf_adaptation_settings_row.addWidget(adapt_iter_label)
        self.tf_adaptation_settings_row.addWidget(self.adapt_iter_spinbox)
        layout.addLayout(self.tf_adaptation_settings_row, 3, 0, 1, 6)

    def _build_tf_attributes(self) -> None:
        tf_settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        self._add_tf_scales_row(tf_settings_layout)
        self._add_tf_use_adaptation_row(tf_settings_layout)
        self._add_tf_adaptation_settings_row(tf_settings_layout)

        self.adapt_checkbox.stateChanged.connect(self._adapt_checkbox_status_changed)

        self.tf_widget = QtWidgets.QWidget()
        self.tf_widget.setLayout(tf_settings_layout)
        self.tf_widget.hide()
        self.main_layout.addWidget(self.tf_widget)

    def _add_torch_use_adaptation_row(self, layout: QtWidgets.QGridLayout):
        # --- Torch section adaptation checkbox with help button ---
        self.torch_adapt_checkbox = QtWidgets.QCheckBox("Use video adaptation")
        self.torch_adapt_checkbox.setChecked(True)
        self.torch_adapt_checkbox.setStyleSheet("font-weight: bold; font-size: 16px; padding: 6px 12px;")
        torch_adapt_help_btn = QtWidgets.QToolButton()
        torch_adapt_help_btn.setIcon(QIcon(os.path.join(BASE_DIR, "assets", "icons", "help2.png")))
        torch_adapt_help_btn.setIconSize(QSize(24, 24))
        torch_adapt_help_btn.setToolTip("What is video adaptation?")
        def show_torch_adapt_help():
            QtWidgets.QMessageBox.information(
                self,
                "Video Adaptation",
                "This will adapt the model on the fly to your video data in a self-supervised way."
            )
        torch_adapt_help_btn.clicked.connect(show_torch_adapt_help)
        use_adaptation_row = QtWidgets.QHBoxLayout()
        use_adaptation_row.addWidget(self.torch_adapt_checkbox)
        use_adaptation_row.addWidget(torch_adapt_help_btn)
        use_adaptation_row.addStretch()
        layout.addLayout(use_adaptation_row, 1, 0, 1, 2)

    def _add_torch_adaptation_settings_row(self, layout: QtWidgets.QGridLayout):
        # Compact adaptation settings row
        pseudo_threshold_label = QtWidgets.QLabel("Pseudo-label confidence threshold")
        pseudo_threshold_label.setMinimumWidth(200)
        self.torch_pseudo_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            decimals=2,
            minimum=0.01,
            maximum=1.0,
            singleStep=0.05,
            value=0.1,
            wrapping=True,
        )
        self.torch_pseudo_threshold_spinbox.setMaximumWidth(100)
        adapt_epoch_label = QtWidgets.QLabel("Number of adaptation epochs")
        adapt_epoch_label.setMinimumWidth(180)
        self.torch_adapt_epoch_spinbox = QtWidgets.QSpinBox()
        self.torch_adapt_epoch_spinbox.setRange(1, 50)
        self.torch_adapt_epoch_spinbox.setValue(4)
        self.torch_adapt_epoch_spinbox.setMaximumWidth(100)
        adapt_det_epoch_label = QtWidgets.QLabel("Number of detector adaptation epochs")
        adapt_det_epoch_label.setMinimumWidth(200)
        self.torch_adapt_det_epoch_spinbox = QtWidgets.QSpinBox()
        self.torch_adapt_det_epoch_spinbox.setRange(1, 50)
        self.torch_adapt_det_epoch_spinbox.setValue(4)
        self.torch_adapt_det_epoch_spinbox.setMaximumWidth(100)
        self.torch_adaptation_settings_row = QtWidgets.QHBoxLayout()
        self.torch_adaptation_settings_row.addWidget(pseudo_threshold_label)
        self.torch_adaptation_settings_row.addWidget(self.torch_pseudo_threshold_spinbox)
        self.torch_adaptation_settings_row.addSpacing(20)
        self.torch_adaptation_settings_row.addWidget(adapt_epoch_label)
        self.torch_adaptation_settings_row.addWidget(self.torch_adapt_epoch_spinbox)
        self.torch_adaptation_settings_row.addSpacing(20)
        self.torch_adaptation_settings_row.addWidget(adapt_det_epoch_label)
        self.torch_adaptation_settings_row.addWidget(self.torch_adapt_det_epoch_spinbox)
        self.torch_adaptation_settings_row.addStretch()
        layout.addLayout(self.torch_adaptation_settings_row, 2, 0, 1, 6)

    def _build_torch_attributes(self) -> None:
        torch_settings_layout = _create_grid_layout(margins=(20, 0, 0, 0))

        self._add_torch_use_adaptation_row(torch_settings_layout)
        self._add_torch_adaptation_settings_row(torch_settings_layout)

        self.torch_adapt_checkbox.stateChanged.connect(
            self._torch_adapt_checkbox_status_changed
        )

        self.torch_widget = QtWidgets.QWidget()
        self.torch_widget.setLayout(torch_settings_layout)
        self.torch_widget.hide()
        self.main_layout.addWidget(self.torch_widget)

    def _adapt_checkbox_status_changed(self, state: int) -> None:
        set_layout_contents_visible(self.tf_adaptation_settings_row, Qt.CheckState(state) == Qt.Checked)

    def _torch_adapt_checkbox_status_changed(self, state: int) -> None:
        set_layout_contents_visible(self.torch_adaptation_settings_row, Qt.CheckState(state) == Qt.Checked)

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
        
        # Show different help based on media type
        media_type = self.media_selection_widget.media_type_widget.currentText()
        if media_type == "Videos":
            help_text = deeplabcut.video_inference_superanimal.__doc__
        else:  # Images
            help_text = deeplabcut.superanimal_analyze_images.__doc__
        
        label = QtWidgets.QLabel(help_text, self)
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

    def run_video_inference_superanimal(self):
        files = list(self.files)
        if not files:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("You must select video files")
            msg.setWindowTitle("Error")
            msg.setMinimumWidth(400)
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec_()
            return

        supermodel_name = self.model_combo.currentText()
        create_labeled_video = self.create_labeled_video_checkbox.isChecked()
        batch_size = int(self.batch_size_combo.currentText())
        detector_batch_size = int(self.detector_batch_size_combo.currentText())
        kwargs = self._gather_kwargs()

        can_run_in_background = False
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("background-color: #9E9E9E; color: white; font-weight: bold;")  # Gray when disabled
        self.root._progress_bar.show()
        try:
            # Use dedicated function for superanimal_humanbody
            if supermodel_name == "superanimal_humanbody":
                # Download config from HuggingFace (needed for the dedicated function)
                from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_snapshot_folder_path
                import huggingface_hub
                
                model_files = get_snapshot_folder_path()
                model_files.mkdir(exist_ok=True)
                
                # Download config file from HuggingFace
                config_path = Path(
                    huggingface_hub.hf_hub_download(
                        "DeepLabCut/HumanBody",
                        "rtmpose-x_simcc-body7_pytorch_config.yaml",
                        local_dir=model_files,
                    )
                )
                
                # Map GUI parameters to dedicated function parameters
                dedicated_kwargs = {
                    "destfolder": self._destfolder,
                    "bbox_threshold": kwargs.get("bbox_threshold", 0.1),
                    "pose_threshold": kwargs.get("pseudo_threshold", 0.4),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "detector_name": kwargs.get("detector_name", "fasterrcnn_mobilenet_v3_large_fpn"),
                }
                
                if can_run_in_background:
                    func = partial(
                        deeplabcut.analyze_videos_superanimal_humanbody,
                        config_path,
                        files,
                        **dedicated_kwargs,
                    )
                    self.worker, self.thread = move_to_separate_thread(func)
                    self.worker.finished.connect(self.signal_analysis_complete)
                    self.thread.start()
                else:
                    print(f"Calling analyze_videos_superanimal_humanbody with config={config_path}, kwargs={dedicated_kwargs}")
                    results = deeplabcut.analyze_videos_superanimal_humanbody(
                        config_path,
                        files,
                        **dedicated_kwargs,
                    )
                    # Patch: Call signal_analysis_complete for non-background execution
                    self.signal_analysis_complete()
            else:
                # Use standard function for other models
                if can_run_in_background:
                    func = partial(
                        deeplabcut.video_inference_superanimal,
                        files,
                        supermodel_name,
                        dest_folder=self._destfolder,
                        create_labeled_video=create_labeled_video,
                        batch_size = batch_size,
                        detector_batch_size = detector_batch_size,
                        **kwargs,
                    )
                    self.worker, self.thread = move_to_separate_thread(func)
                    self.worker.finished.connect(self.signal_analysis_complete)
                    self.thread.start()
                else:
                    print(f"Calling video_inference_superanimal with kwargs={kwargs}")
                    results = deeplabcut.video_inference_superanimal(
                        files,
                        supermodel_name,
                        dest_folder=self._destfolder,
                        create_labeled_video=create_labeled_video,
                        batch_size = batch_size,
                        detector_batch_size = detector_batch_size,
                        **kwargs,
                    )
                # Check for skipped frames and show warning if needed
                for video_path in files:
                    try:
                        df = results[video_path]
                        n_processed = len(df)
                        cap = cv2.VideoCapture(video_path)
                        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if n_processed < n_total:
                            msg = QtWidgets.QMessageBox()
                            msg.setIcon(QtWidgets.QMessageBox.Warning)
                            msg.setText(f"Warning: Only {n_processed} out of {n_total} frames had detections. The output movie and results include only those frames.")
                            msg.setWindowTitle("Partial Detections")
                            msg.setMinimumWidth(400)
                            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                            msg.exec_()
                    except Exception as e:
                        print(f"[GUI Warning] Could not check processed frames: {e}")
                self.signal_analysis_complete()
        except Exception as e:
            print(f"[Error] {e}")
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")  # Green when enabled
            self.root._progress_bar.hide()

    def signal_analysis_complete(self):
        self.run_button.setEnabled(True)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")  # Green when enabled
        self.root._progress_bar.hide()
        
        # Check if labeled videos were actually created
        files = list(self.files)
        videos_created = []
        
        # Determine the output folder
        output_folder = self._destfolder if self._destfolder else Path(files[0]).parent
        
        for video_path in files:
            video_name = Path(video_path).stem
            if self.model_combo.currentText() == "superanimal_humanbody":
                # Check for humanbody labeled video - use the pattern that the dedicated function creates
                # The dedicated function creates: {video_name}_{dlc_scorer}_labeled.mp4
                # We need to look for files that match this pattern
                labeled_videos = list(Path(output_folder).glob(f"{video_name}_*_labeled.mp4"))
                if labeled_videos:
                    videos_created.extend([str(v) for v in labeled_videos])
            else:
                # Check for standard labeled video - use the pattern that the standard function creates
                # The standard function creates: {video_name}_{dlc_scorer}_labeled.mp4
                # For video adaptation, it also creates: {video_name}_{dlc_scorer}_labeled_before_adapt.mp4 and _after_adapt.mp4
                # We need to look for files that match any of these patterns
                labeled_videos = list(Path(output_folder).glob(f"{video_name}_*_labeled*.mp4"))
                if labeled_videos:
                    videos_created.extend([str(v) for v in labeled_videos])
        
        # Show appropriate message
        media_type = self.media_selection_widget.media_type_widget.currentText()
        if videos_created:
            msg = QtWidgets.QMessageBox(text=f"SuperAnimal {media_type.lower()} inference complete!\n\nCreated labeled videos:\n" + "\n".join(videos_created))
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.exec_()
        else:
            msg = QtWidgets.QMessageBox(text=f"SuperAnimal {media_type.lower()} inference complete, but no labeled videos were created.")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    def stop_processes(self):
        """Stop any running processes"""
        if self.thread and self.thread.isRunning():
            print("Stopping running processes...")
            self.thread.quit()
            self.thread.wait(5000)  # Wait up to 5 seconds
            if self.thread.isRunning():
                self.thread.terminate()
                self.thread.wait(2000)
            self.worker = None
            self.thread = None
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.root._progress_bar.hide()

    def closeEvent(self, event):
        """Override closeEvent to stop processes when tab is closed"""
        self.stop_processes()
        super().closeEvent(event)

    def _update_adaptation_options(self, media_type: str):
        # Only allow video adaptation for videos (no images)
        self.adapt_checkbox.setEnabled(True)
        self.torch_adapt_checkbox.setEnabled(True)
        self.run_button.setText("Run")

    def _gather_kwargs(self) -> dict:
        kwargs = dict(model_name=self.net_type_selector.currentText())
        media_type = self.media_selection_widget.media_type_widget.currentText()
        
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
            # Only allow video adaptation for videos
            if media_type == "Videos":
                kwargs["video_adapt"] = self.adapt_checkbox.isChecked()
            else:
                kwargs["video_adapt"] = False
            kwargs["pseudo_threshold"] = self.pseudo_threshold_spinbox.value()
            kwargs["adapt_iterations"] = self.adapt_iter_spinbox.value()
        else:
            kwargs["detector_name"] = self.detector_type_selector.currentText()
            # Only allow video adaptation for videos
            if media_type == "Videos":
                kwargs["video_adapt"] = self.torch_adapt_checkbox.isChecked()
            else:
                kwargs["video_adapt"] = False
            kwargs["pseudo_threshold"] = self.pose_threshold_spinbox.value()
            kwargs["bbox_threshold"] = self.detector_threshold_spinbox.value()
            kwargs["detector_epochs"] = self.torch_adapt_det_epoch_spinbox.value()
            kwargs["pose_epochs"] = self.torch_adapt_epoch_spinbox.value()
            kwargs["max_individuals"] = self.max_individuals_spinbox.value()

        return kwargs

    def _update_available_models(self, engine: Engine) -> None:
        current_dataset = self.model_combo.currentText()

        if engine == Engine.TF:
            supermodels = ["superanimal_topviewmouse", "superanimal_quadruped"]
        else:
            supermodels = dlclibrary.get_available_datasets()

        set_combo_items(
            combo_box = self.model_combo,
            items = supermodels,
            index = supermodels.index(current_dataset) if current_dataset in supermodels else 0,
        )

    def _update_pose_models(self, super_animal: str) -> None:
        if len(super_animal) == 0:
            set_combo_items(
                combo_box = self.net_type_selector,
                items = []
            )
            return

        set_combo_items(
            combo_box = self.net_type_selector,
            items = ["dlcrnet"] if self.root.engine == Engine.TF else dlclibrary.get_available_models(super_animal)
        )

    def _update_detectors(self, super_animal: str) -> None:
        if len(super_animal) == 0:
            set_combo_items(
                combo_box = self.detector_type_selector,
                items = []
            )
            return

        set_combo_items(
            combo_box = self.detector_type_selector,
            items = [] if self.root.engine == Engine.TF else dlclibrary.get_available_detectors(super_animal)
        )
        set_layout_contents_visible(self.detector_row, self.root.engine == Engine.PYTORCH and super_animal != "superanimal_humanbody")

    def _update_adaptation_visibility(self, super_animal: str):
        if self.root.engine == Engine.PYTORCH and super_animal != "superanimal_humanbody":
            self.torch_widget.show()
        else:
            self.torch_widget.hide()

    @Slot(Engine)
    def _on_engine_change(self, engine: Engine) -> None:
        self._update_available_models(engine)
        if engine == Engine.PYTORCH:
            self.tf_widget.hide()
            self.torch_widget.show()
        else:
            self.torch_widget.hide()
            self.tf_widget.show()

        # Hide widgets in detector row
        set_layout_contents_visible(self.detector_row, engine == Engine.PYTORCH and self.model_combo.currentText() != "superanimal_humanbody")

        # Initialize adaptation options based on current media type
        current_media_type = self.media_selection_widget.media_type_widget.currentText()
        self._update_adaptation_options(current_media_type)
