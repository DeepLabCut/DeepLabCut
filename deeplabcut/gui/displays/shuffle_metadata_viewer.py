#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Widget to display existing shuffles"""
from __future__ import annotations

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

import deeplabcut.generate_training_dataset.metadata as metadata


class ShuffleMetadataViewer(QtWidgets.QDialog):
    """Viewer for shuffle metadata"""

    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.root = root
        self.parent = parent
        self.file_content = _load_metadata(self.root.cfg)

        self.setWindowTitle("Existing Shuffles: Metadata")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        inner_layout = QtWidgets.QVBoxLayout()
        inner_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        inner_layout.setSpacing(0)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        for line in self.file_content:

            inner_layout.addWidget(QtWidgets.QLabel(line))

        inner = QtWidgets.QFrame(scroll)
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)


def _load_metadata(cfg: dict) -> list[str]:
    metadata_path = metadata.TrainingDatasetMetadata.path(cfg)
    if not metadata_path.exists():
        trainset_meta = metadata.TrainingDatasetMetadata.create(cfg)
        trainset_meta.save()

    with open(metadata_path, "r") as file:
        raw_metadata = file.read()

    return raw_metadata.split("\n")
