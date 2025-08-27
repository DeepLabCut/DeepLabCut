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
from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from deeplabcut.generate_training_dataset import check_labels
from deeplabcut.gui.components import DefaultTab
from deeplabcut.gui.widgets import launch_napari
from deeplabcut.utils.skeleton import SkeletonBuilder


def label_frames(
    config_path: str | Path | None = None,
    image_folder: str | None = None
):
    """Launches the napari-deeplabcut labelling GUI.

    For more information on labelling data with napari-deeplabcut, see our docs:
        https://github.com/DeepLabCut/napari-deeplabcut?tab=readme-ov-file#usage

    If no parameters are given, the napari-deeplabcut labelling GUI is simply open,
    and the folder containing the images to label can be dropped into the GUI.

    If the `config_path` and the `image_folder` are given as arguments, the given
    `image_folder` for the project is opened in the napari-deeplabcut GUI to be labeled.
    If only the `config_path` is given, the first image folder is opened.

    Parameters
    ----------
    config_path: str, Path, None
        Full path of the project config.yaml file.

    image_folder: str, None
        Name of the image folder to open for labelling.

    Examples
    --------
    Opening the napari-deeplabcut annotation GUI without opening a specific folder of
    images to label. You then need to drag-and-drop your image folder into the GUI.
    See the napari-deeplabcut docs linked above for more information about labelling in
    napari-deeplabcut.
    >>> import deeplabcut
    >>> deeplabcut.label_frames()

    Opening the images extracted from the "2025-01-01-experiment7" video in
    napari-deeplabcut on Windows. The project's folder structure should look as follows:
    reaching-task/                    # project root directory
    ├── config.yaml                   # project configuration file
    └── labeled-data/                 # folder containing all extracted image folders
        ├── ...
        ├── 2025-01-01-experiment7    # folder containing the images to label
        └── ...

    >>> deeplabcut.label_frames(
    >>>     "C:\\myproject\\reaching-task\\config.yaml",
    >>>     "2025-01-01-experiment7",
    >>> )

    Opening the images extracted from the first video listed in the project
    configuration in napari-deeplabcut on a Unix system.
    >>> deeplabcut.label_frames("/users/john/project/config.yaml")
    """
    files = None
    if config_path is None:
        if image_folder is not None:
            raise ValueError(
                f"If the ``config_path`` is None, the ``image_folder`` must be None "
                f"too. Found {image_folder}. To label the images in {image_folder}, "
                f"give the project configuration file as `config_path`."
            )
    else:
        data_dir = Path(config_path).parent / "labeled-data"
        if image_folder is None:
            image_dirs = [path for path in data_dir.iterdir() if path.is_dir()]
            if len(image_dirs) == 0:
                raise ValueError(
                    f"Could not find any image folders in {data_dir}. Please check "
                    f"the config path given to `deeplabcut.label_frames(...)`"
                )
            image_dir = list(sorted(image_dirs))[0]
        else:
            image_dir = data_dir / image_folder

        files = [str(image_dir), str(config_path)]
    _ = launch_napari(files=files)


refine_labels = label_frames


class LabelFrames(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(LabelFrames, self).__init__(root, parent, h1_description)

        self._set_page()

    def _set_page(self):
        self.label_frames_btn = QtWidgets.QPushButton("Label Frames")
        self.label_frames_btn.clicked.connect(self.label_frames)
        self.check_labels_btn = QtWidgets.QPushButton("Check Labels")
        self.check_labels_btn.clicked.connect(self.check_labels)
        self.build_skeleton_btn = QtWidgets.QPushButton("Build skeleton")
        self.build_skeleton_btn.clicked.connect(self.build_skeleton)
        self.main_layout.addWidget(self.label_frames_btn, alignment=Qt.AlignLeft)
        self.main_layout.addWidget(self.check_labels_btn, alignment=Qt.AlignLeft)
        self.main_layout.addWidget(self.build_skeleton_btn, alignment=Qt.AlignLeft)

    def log_color_by_option(self, choice):
        self.root.logger.info(f"Labeled images will by colored by {choice.upper()}")

    def label_frames(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        dialog.setDirectory(
            os.path.join(os.path.dirname(self.root.config), "labeled-data")
        )
        if dialog.exec_():
            folder = dialog.selectedFiles()[0]
            has_h5 = False
            for file in os.listdir(folder):
                if file.endswith(".h5"):
                    has_h5 = True
                    break
            if not has_h5:
                folder = [folder, self.root.config]
            _ = launch_napari(folder)

    def check_labels(self):
        check_labels(self.root.config, visualizeindividuals=self.root.is_multianimal)
        labeled_images = (Path(self.root.config).parent / "labeled-data").rglob("*_labeled/*.png")
        _ = launch_napari(labeled_images, plugin="napari", stack=True)

    def build_skeleton(self, *args):
        SkeletonBuilder(self.root.config)