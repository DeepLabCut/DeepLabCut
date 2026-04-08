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
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import deeplabcut
from deeplabcut.gui.components import (
    BodypartListWidget,
    DefaultTab,
    ShuffleSpinBox,
    VideoSelectionWidget,
    _create_grid_layout,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
from deeplabcut.gui.utils import move_to_separate_thread
from deeplabcut.gui.widgets import ConfigEditor
from deeplabcut.utils.auxiliaryfunctions import edit_config


@dataclass(frozen=True)
class AnalyzeVideosOptions:
    config: str
    shuffle: int
    save_as_csv: bool
    filter_data: bool
    plot_trajectories: bool
    show_trajectory_plots: bool
    displayed_bodyparts: tuple[str, ...]
    create_video_all_detections: bool
    auto_track: bool
    calibrate_assembly: bool
    assemble_with_ID_only: bool
    num_animals_in_videos: int | None
    cropping: tuple[int, int, int, int] | None
    dynamic_cropping_params: tuple[bool, float, int]
    track_method: str | None


class AnalyzeVideos(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super().__init__(root, parent, h1_description)

        self._set_page()

    @property
    def files(self):
        return self.video_selection_widget.files

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Video Selection", "font:bold"))
        self.video_selection_widget = VideoSelectionWidget(
            self.root, self, hide_videotype=True, sync_videotype_with_selection=True
        )
        self.main_layout.addWidget(self.video_selection_widget)

        tmp_layout = _create_horizontal_layout()

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout()
        self._generate_layout_attributes(self.layout_attributes)
        tmp_layout.addLayout(self.layout_attributes)

        # Single / Multi animal Only Layouts
        self.layout_singleanimal = _create_horizontal_layout()
        self.layout_multianimal = _create_horizontal_layout()

        if self.root.is_multianimal:
            self._generate_layout_multianimal(self.layout_multianimal)
            tmp_layout.addLayout(self.layout_multianimal)
        else:
            self._generate_layout_single_animal(self.layout_singleanimal)
            tmp_layout.addLayout(self.layout_singleanimal)

        self.main_layout.addLayout(tmp_layout)

        self.main_layout.addWidget(_create_label_widget("", "font:bold"))
        self.layout_other_options = _create_vertical_layout()
        self._generate_layout_other_options(self.layout_other_options)
        self.main_layout.addLayout(self.layout_other_options)

        self.analyze_videos_btn = QtWidgets.QPushButton("Analyze Videos")
        self.analyze_videos_btn.clicked.connect(self.analyze_videos)

        self.edit_config_file_btn = QtWidgets.QPushButton("Edit config.yaml")
        self.edit_config_file_btn.clicked.connect(self.edit_config_file)

        self.main_layout.addWidget(self.analyze_videos_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.edit_config_file_btn, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(deeplabcut.analyze_videos.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_single_animal(self, layout):
        # Dynamic bodypart cropping
        self.crop_bodyparts = QtWidgets.QCheckBox("Dynamically crop bodyparts")
        self.crop_bodyparts.setCheckState(Qt.Unchecked)
        self.crop_bodyparts.stateChanged.connect(self.update_crop_choice)

        self.dynamic_cropping = False
        layout.addWidget(self.crop_bodyparts)

    def _generate_layout_other_options(self, layout):
        tmp_layout = _create_horizontal_layout(margins=(0, 0, 0, 0))

        # Save results as csv
        self.save_as_csv = QtWidgets.QCheckBox("Save result(s) as csv")
        self.save_as_csv.setCheckState(Qt.Unchecked)
        self.save_as_csv.stateChanged.connect(self.update_csv_choice)

        tmp_layout.addWidget(self.save_as_csv)

        # Filter predictions
        self.filter_predictions = QtWidgets.QCheckBox("Filter predictions")
        self.filter_predictions.setCheckState(Qt.Unchecked)
        self.filter_predictions.stateChanged.connect(self.update_filter_choice)

        tmp_layout.addWidget(self.filter_predictions)

        # Plot Trajectories
        self.plot_trajectories = QtWidgets.QCheckBox("Plot trajectories")
        self.plot_trajectories.setCheckState(Qt.Unchecked)
        self.plot_trajectories.stateChanged.connect(self.update_plot_trajectory_choice)

        tmp_layout.addWidget(self.plot_trajectories)

        # Show trajectory plots
        self.show_trajectory_plots = QtWidgets.QCheckBox("Show trajectory plots")
        self.show_trajectory_plots.setCheckState(Qt.Unchecked)
        self.show_trajectory_plots.setEnabled(False)
        self.show_trajectory_plots.stateChanged.connect(self.update_showfigs_choice)

        tmp_layout.addWidget(self.show_trajectory_plots)

        layout.addLayout(tmp_layout)

        self.bodyparts_list_widget = BodypartListWidget(root=self.root, parent=self)
        layout.addWidget(self.bodyparts_list_widget, Qt.AlignLeft)

    def _generate_layout_attributes(self, layout):
        # Shuffle
        opt_text = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        layout.addWidget(opt_text, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)

    def _generate_layout_multianimal(self, layout):
        tmp_layout = QtWidgets.QGridLayout()

        opt_text = QtWidgets.QLabel("Tracking method")
        self.tracker_type_widget = QtWidgets.QComboBox()
        self.tracker_type_widget.addItems(["ellipse", "box", "skeleton"])
        self.tracker_type_widget.currentTextChanged.connect(self.update_tracker_type)
        tmp_layout.addWidget(opt_text, 0, 0)
        tmp_layout.addWidget(self.tracker_type_widget, 0, 1)

        opt_text = QtWidgets.QLabel("Number of animals in videos")
        self.num_animals_in_videos = QtWidgets.QSpinBox()
        self.num_animals_in_videos.setMaximum(100)
        self.num_animals_in_videos.setValue(len(self.root.all_individuals))
        tmp_layout.addWidget(opt_text, 1, 0)
        tmp_layout.addWidget(self.num_animals_in_videos, 1, 1)

        # layout.addLayout(tmp_layout)

        # tmp_layout = QtWidgets.QGridLayout()

        self.calibrate_assembly_checkbox = QtWidgets.QCheckBox("Calibrate assembly")
        self.calibrate_assembly_checkbox.setCheckState(Qt.Unchecked)
        self.calibrate_assembly_checkbox.stateChanged.connect(self.update_calibrate_assembly)
        tmp_layout.addWidget(self.calibrate_assembly_checkbox, 0, 2)

        self.assemble_with_ID_only_checkbox = QtWidgets.QCheckBox("Assemble with ID only")
        self.assemble_with_ID_only_checkbox.setCheckState(Qt.Unchecked)
        self.assemble_with_ID_only_checkbox.stateChanged.connect(self.update_assemble_with_ID_only)
        tmp_layout.addWidget(self.assemble_with_ID_only_checkbox, 0, 3)

        self.create_detections_video_checkbox = QtWidgets.QCheckBox("Create video with all detections")
        self.create_detections_video_checkbox.setCheckState(Qt.Unchecked)
        self.create_detections_video_checkbox.stateChanged.connect(self.update_create_video_detections)
        tmp_layout.addWidget(self.create_detections_video_checkbox, 0, 4)

        layout.addLayout(tmp_layout)

    def update_create_video_detections(self, state):
        s = "ENABLED" if Qt.CheckState(state) == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Create video with all detections {s}")

    def update_assemble_with_ID_only(self, state):
        s = "ENABLED" if Qt.CheckState(state) == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Assembly with ID only {s}")

    def update_calibrate_assembly(self, state):
        s = "ENABLED" if Qt.CheckState(state) == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Assembly calibration {s}")

    def update_tracker_type(self, method):
        self.root.logger.info(f"Using {method.upper()} tracker")

    def update_csv_choice(self, state):
        s = "ENABLED" if Qt.CheckState(state) == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Save results as CSV {s}")

    def update_filter_choice(self, state):
        s = "ENABLED" if Qt.CheckState(state) == Qt.Checked else "DISABLED"
        self.root.logger.info(f"Filtering predictions {s}")

    def update_showfigs_choice(self, state):
        if Qt.CheckState(state) == Qt.Checked:
            self.root.logger.info("Plots will show as pop ups.")
        else:
            self.root.logger.info("Plots will not show up.")

    def update_crop_choice(self, state):
        if Qt.CheckState(state) == Qt.Checked:
            self.root.logger.info("Dynamic bodypart cropping ENABLED.")
            self.dynamic_cropping = True
        else:
            self.root.logger.info("Dynamic bodypart cropping DISABLED.")
            self.dynamic_cropping = False

    def update_plot_trajectory_choice(self, state):
        if Qt.CheckState(state) == Qt.Checked:
            self.bodyparts_list_widget.refresh()
            self.bodyparts_list_widget.show()
            self.bodyparts_list_widget.setEnabled(True)
            self.show_trajectory_plots.setEnabled(True)
            self.root.logger.info("Plot trajectories ENABLED.")

        else:
            self.bodyparts_list_widget.hide()
            self.bodyparts_list_widget.setEnabled(False)
            self.show_trajectory_plots.setEnabled(False)
            self.show_trajectory_plots.setCheckState(Qt.Unchecked)
            self.root.logger.info("Plot trajectories DISABLED.")

    def edit_config_file(self):
        if not self.root.config:
            return
        editor = ConfigEditor(self.root.config)
        editor.show()

    def _collect_options(self) -> AnalyzeVideosOptions:
        config = self.root.config
        shuffle = self.root.shuffle_value
        save_as_csv = self.save_as_csv.isChecked()
        filter_data = self.filter_predictions.isChecked()
        plot_trajectories = self.plot_trajectories.isChecked()
        show_trajectory_plots = self.show_trajectory_plots.isChecked()
        displayed_bodyparts = tuple(self.bodyparts_list_widget.selected_bodyparts) if plot_trajectories else ()

        if self.root.is_multianimal:
            calibrate_assembly = self.calibrate_assembly_checkbox.isChecked()
            assemble_with_ID_only = self.assemble_with_ID_only_checkbox.isChecked()
            track_method = self.tracker_type_widget.currentText()
            num_animals_in_videos = self.num_animals_in_videos.value()
            create_video_all_detections = self.create_detections_video_checkbox.isChecked()
        else:
            calibrate_assembly = False
            assemble_with_ID_only = False
            track_method = None
            num_animals_in_videos = None
            create_video_all_detections = False

        cropping = None
        crop_flag = self.root.cfg.get("cropping", False)
        if str(crop_flag).lower() == "true":
            cropping = (
                self.root.cfg["x1"],
                self.root.cfg["x2"],
                self.root.cfg["y1"],
                self.root.cfg["y2"],
            )

        dynamic_cropping_params = (False, 0.5, 10)
        if getattr(self, "dynamic_cropping", False):
            dynamic_cropping_params = (True, 0.5, 10)

        return AnalyzeVideosOptions(
            config=config,
            shuffle=shuffle,
            save_as_csv=save_as_csv,
            filter_data=filter_data,
            plot_trajectories=plot_trajectories,
            show_trajectory_plots=show_trajectory_plots,
            displayed_bodyparts=displayed_bodyparts,
            create_video_all_detections=create_video_all_detections,
            auto_track=self.root.is_multianimal,
            calibrate_assembly=calibrate_assembly,
            assemble_with_ID_only=assemble_with_ID_only,
            num_animals_in_videos=num_animals_in_videos,
            cropping=cropping,
            dynamic_cropping_params=dynamic_cropping_params,
            track_method=track_method,
        )

    def _get_video_batches(self):
        """
        Returns a list of (videotype, videos) pairs.
        videotype should include the leading dot, e.g. '.avi'.
        """
        groups = self.video_selection_widget.get_files_grouped_by_suffix(keep_dot=True)
        batches = [(suffix, videos) for suffix, videos in sorted(groups.items()) if suffix]
        return batches

    def _get_unique_video_parent_folders(self, batches: list[tuple[str, list[str]]]) -> list[str]:
        folders = []
        seen = set()

        for _, videos in batches:
            for video in videos:
                parent = str(Path(video).parent.resolve())
                if parent not in seen:
                    seen.add(parent)
                    folders.append(parent)

        return folders

    def _run_pipeline(self, options: AnalyzeVideosOptions, batches: list[tuple[str, list[str]]]):
        for videotype, videos in batches:
            try:
                self.root.logger.info(f"Analyzing {len(videos)} video(s) with extension {videotype}")

                deeplabcut.analyze_videos(
                    options.config,
                    videos=videos,
                    videotype=videotype,
                    shuffle=options.shuffle,
                    save_as_csv=options.save_as_csv,
                    cropping=options.cropping,
                    dynamic=options.dynamic_cropping_params,
                    auto_track=options.auto_track,
                    n_tracks=options.num_animals_in_videos,
                    calibrate=options.calibrate_assembly,
                    identity_only=options.assemble_with_ID_only,
                )

                self._run_postprocessing_for_group(options, videotype, videos)
            except Exception as e:
                exc = f"Error analyzing videos {videos} with extension {videotype}: {e}"
                self.root.logger.error(exc, exc_info=True)
                raise RuntimeError(exc) from e

        # Run CSV conversion once per unique folder, after all batches
        if options.auto_track and options.save_as_csv:
            self._convert_outputs_to_csv_once_per_folder(batches)

    def _run_postprocessing_for_group(
        self,
        options: AnalyzeVideosOptions,
        videotype: str,
        videos: list[str],
    ):
        if options.create_video_all_detections:
            deeplabcut.create_video_with_all_detections(
                options.config,
                videos=videos,
                videotype=videotype,
                shuffle=options.shuffle,
            )

        if options.filter_data:
            deeplabcut.filterpredictions(
                options.config,
                video=videos,
                videotype=videotype,
                shuffle=options.shuffle,
                filtertype="median",
                windowlength=5,
                save_as_csv=options.save_as_csv,
                track_method=options.track_method,
            )

        if options.plot_trajectories:
            deeplabcut.plot_trajectories(
                options.config,
                videos=videos,
                displayedbodyparts=options.displayed_bodyparts,
                videotype=videotype,
                shuffle=options.shuffle,
                filtered=options.filter_data,
                showfigures=options.show_trajectory_plots,
                track_method=options.track_method,
            )

    def _convert_outputs_to_csv_once_per_folder(self, batches: list[tuple[str, list[str]]]):
        folders = self._get_unique_video_parent_folders(batches)

        for folder in folders:
            self.root.logger.info(f"Converting H5 outputs to CSV in folder: {folder}")
            deeplabcut.analyze_videos_converth5_to_csv(
                folder,
                listofvideos=False,
            )

    def analyze_videos(self):
        options = self._collect_options()
        batches = self._get_video_batches()

        if not batches:
            self.root.logger.warning("No videos selected.")
            return

        # Keep config in sync with GUI choice before launching worker
        if self.root.is_multianimal and options.track_method is not None:
            edit_config(self.root.config, {"default_track_method": options.track_method})

        func = partial(self._run_pipeline, options, batches)

        self.worker, self.thread = move_to_separate_thread(func)
        self.worker.finished.connect(lambda: self.analyze_videos_btn.setEnabled(True))
        self.worker.finished.connect(lambda: self.root._progress_bar.hide())
        self.thread.start()
        self.analyze_videos_btn.setEnabled(False)
        self.root._progress_bar.show()
