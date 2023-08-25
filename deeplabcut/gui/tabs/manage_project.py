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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
)
from deeplabcut.create_project import add_new_videos
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.components import DefaultTab, _create_horizontal_layout
from deeplabcut.gui.widgets import ConfigEditor


class ManageProject(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super().__init__(root, parent, h1_description)
        self._set_page()
        self._videos = []

    def _set_page(self):
        # Add config text field and button
        project_config_layout = _create_horizontal_layout()

        cfg_text = QLabel("Active config file:")

        self.cfg_line = QLineEdit()
        self.cfg_line.setText(self.root.config)
        self.cfg_line.textChanged[str].connect(self.root.update_cfg)

        browse_button = QPushButton("Browse")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.root._open_project)

        project_config_layout.addWidget(cfg_text)
        project_config_layout.addWidget(self.cfg_line)
        project_config_layout.addWidget(browse_button)

        self.main_layout.addLayout(project_config_layout)

        self.edit_btn = QPushButton("Edit config.yaml")
        self.edit_btn.setMinimumWidth(150)
        self.edit_btn.clicked.connect(self.open_config_editor)

        self.add_videos_btn = QPushButton("Add new videos")
        self.add_videos_btn.clicked.connect(self.add_new_videos)

        self.main_layout.addWidget(self.edit_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.add_videos_btn, alignment=Qt.AlignRight)

    def open_config_editor(self):
        editor = ConfigEditor(self.root.config)
        editor.show()

    def add_new_videos(self):
        cwd = os.getcwd()
        files = QFileDialog.getOpenFileNames(
            self,
            "Select videos to add to the project",
            cwd,
            f"Videos ({' *.'.join(DLCParams.VIDEOTYPES)[1:]})",
        )[0]
        if not files:
            return

        add_new_videos(self.root.config, files)
