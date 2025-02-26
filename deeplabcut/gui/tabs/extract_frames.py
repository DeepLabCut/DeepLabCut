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
from functools import partial
from pathlib import Path
from typing import Union

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.components import (
    DefaultTab,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.utils import move_to_separate_thread
from deeplabcut.gui.widgets import launch_napari
from deeplabcut.generate_training_dataset import extract_frames


def select_cropping_area(config, videos=None):
    """
    Interactively select the cropping area of all videos in the config.
    A user interface pops up with a frame to select the cropping parameters.
    Use the left click to draw a box and hit the button 'set cropping parameters'
    to store the cropping parameters for a video in the config.yaml file.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : optional (default=None)
        List of videos whose cropping areas are to be defined. Note that full paths are required.
        By default, all videos in the config are successively loaded.

    Returns
    -------
    cfg : dict
        Updated project configuration
    """
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.gui.widgets import FrameCropper

    cfg = auxiliaryfunctions.read_config(config)
    if videos is None:
        videos = list(cfg.get("video_sets_original") or cfg["video_sets"])

    for video in videos:
        fc = FrameCropper(video)
        coords = fc.draw_bbox()
        if coords:
            temp = {
                "crop": ", ".join(
                    map(
                        str,
                        [
                            int(coords[0]),
                            int(coords[2]),
                            int(coords[1]),
                            int(coords[3]),
                        ],
                    )
                )
            }
            try:
                cfg["video_sets"][video] = temp
            except KeyError:
                cfg["video_sets_original"][video] = temp

    auxiliaryfunctions.write_config(config, cfg)
    return cfg


class ExtractFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(ExtractFrames, self).__init__(root, parent, h1_description)
        self.worker = None
        self.thread = None
        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(0, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(
            _create_label_widget(
                "Frame extraction from a video subset (optional for automatic extraction)",
                "font:bold",
            )
        )
        self.video_selection_widget = VideoSelectionWidget(self.root, self)
        self.main_layout.addWidget(self.video_selection_widget)

        self.ok_button = QtWidgets.QPushButton("Extract Frames")
        self.ok_button.clicked.connect(self.extract_frames)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(extract_frames.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(1, 300)
        # Extraction method
        ext_method_label = QtWidgets.QLabel("Extraction method")
        self.extraction_method_widget = QtWidgets.QComboBox()
        options = ["automatic", "manual"]
        self.extraction_method_widget.addItems(options)
        self.extraction_method_widget.currentTextChanged.connect(
            self.log_extraction_method
        )

        # Frame extraction algorithm
        ext_algo_label = QtWidgets.QLabel("Extraction algorithm")
        self.extraction_algorithm_widget = QtWidgets.QComboBox()
        self.extraction_algorithm_widget.addItems(DLCParams.FRAME_EXTRACTION_ALGORITHMS)
        self.extraction_algorithm_widget.currentTextChanged.connect(
            self.log_extraction_algorithm
        )

        # Frame cropping
        frame_crop_label = QtWidgets.QLabel("Frame cropping")
        self.frame_cropping_widget = QtWidgets.QComboBox()
        self.frame_cropping_widget.addItems(["disabled", "read from config", "GUI"])
        self.frame_cropping_widget.currentTextChanged.connect(
            self.log_frame_cropping_choice
        )

        # Cluster step
        cluster_step_label = QtWidgets.QLabel("Cluster step")
        self.cluster_step_widget = QtWidgets.QSpinBox()
        self.cluster_step_widget.setValue(1)

        # GUI Slider width
        gui_slider_label = QtWidgets.QLabel("GUI slider width")
        self.slider_width_widget = QtWidgets.QSpinBox()
        self.slider_width_widget.setValue(25)
        self.slider_width_widget.setEnabled(False)

        layout.addWidget(ext_method_label, 1, 0)
        layout.addWidget(self.extraction_method_widget, 1, 1)
        layout.addWidget(gui_slider_label, 1, 2)
        layout.addWidget(self.slider_width_widget, 1, 3)

        layout.addWidget(ext_algo_label, 2, 0)
        layout.addWidget(self.extraction_algorithm_widget, 2, 1)
        layout.addWidget(cluster_step_label, 2, 2)
        layout.addWidget(self.cluster_step_widget, 2, 3)

        layout.addWidget(frame_crop_label, 3, 0)
        layout.addWidget(self.frame_cropping_widget, 3, 1)

    def log_extraction_algorithm(self, extraction_algorithm):
        self.root.logger.info(f"Extraction method set to {extraction_algorithm}")

    def log_extraction_method(self, extraction_method):
        self.root.logger.info(f"Extraction method set to {extraction_method}")
        if extraction_method == "manual":
            self.extraction_algorithm_widget.setEnabled(False)
            self.cluster_step_widget.setEnabled(False)
            self.frame_cropping_widget.setEnabled(False)
            self.slider_width_widget.setEnabled(True)
        else:
            self.extraction_algorithm_widget.setEnabled(True)
            self.cluster_step_widget.setEnabled(True)
            self.frame_cropping_widget.setEnabled(True)
            self.slider_width_widget.setEnabled(False)

    def log_frame_cropping_choice(self, cropping_option):
        self.root.logger.info(f"Cropping set to '{cropping_option}'")

    def extract_frames(self):
        config = self.root.config
        mode = self.extraction_method_widget.currentText()
        if mode == "manual":
            videos = list(self.video_selection_widget.files)
            if not videos:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    "Please select exactly one video to extract frames from.",
                )
                return
            first_video = videos[0]
            if len(videos) > 1:
                self.root.writer.write(
                    f"Only the first video ({first_video}) will be opened."
                )
            video_path_in_folder = self._check_symlink(first_video)
            _ = launch_napari(str(video_path_in_folder))
            return

        algo = self.extraction_algorithm_widget.currentText()
        clusterstep = self.cluster_step_widget.value()
        slider_width = self.slider_width_widget.value()

        crop = False  # default value
        if self.frame_cropping_widget.currentText() == "GUI":
            _ = select_cropping_area(config)
            crop = True
        elif self.frame_cropping_widget.currentText() == "read from config":
            crop = True

        func = partial(
            extract_frames,
            config,
            mode,
            algo,
            crop=crop,
            cluster_step=clusterstep,
            cluster_resizewidth=30,
            cluster_color=False,
            slider_width=slider_width,
            userfeedback=False,
            videos_list=self.video_selection_widget.files or None,
        )

        self.worker, self.thread = move_to_separate_thread(func, capture_outputs=True)
        self.worker.finished.connect(lambda: self.ok_button.setEnabled(True))
        self.worker.finished.connect(lambda: self.root._progress_bar.hide())
        self.thread.finished.connect(self._show_success_message)
        self.thread.start()
        self.ok_button.setEnabled(False)
        self.root._progress_bar.show()

    def _show_success_message(self):
        message = "Failed to create worker: it is None"
        root_message = "failed to extract frames: worker is None"
        if self.worker is not None:
            failed = self.worker.outputs
            if failed is None:
                # outputs are None during manual frame extraction
                return

            if len(failed) == 0:
                message = (
                    "Frame extraction failed. Please check your terminal output "
                    "for more information."
                )
            elif all(failed):
                message = "Frame extraction failed. Video files must be corrupted."
            elif any(failed):
                message = "Although most frames were extracted, some were invalid."
                root_message = "failed to extract (some) frames"
            else:
                message = (
                    "Frames were successfully extracted, for the videos of interest."
                )
                root_message = "successfully extracted frames"

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Info")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        self.root.writer.write(root_message)

    def _check_symlink(self, video_path: Union[str, Path]) -> Path:
        """Checks that a video is in the DeepLabCut 'videos' folder

        This is required before launching manual frame extraction. When users select
        a symlink of a video using the VideoSelectionWidget, the path is resolved to the
        true path of the video (which leads napari-deeplabcut to save the frames in the
        incorrect folder).

        Args:
            video_path: the path to a video in a DeepLabCut project or a video that was
                added to the project

        Returns:
            the path to the video (or symlink) in the project's 'videos' folder

        Raises:
            FileNotFoundError if there is no symlink or video in the 'videos' folder for
                the given video
        """
        video_path = Path(video_path).resolve()
        project_videos = (Path(self.root.config).parent / "videos").resolve()
        if video_path.parent == project_videos:
            return video_path

        symlink_path = project_videos / video_path.name
        if not symlink_path.exists():
            raise FileNotFoundError(
                f"Could not find the video {video_path.name} in your project videos. "
                f"Did you add the video (you can do so in the 'Manage Project' tab)? "
                f"There should be a file in {symlink_path}."
            )

        return symlink_path
