import os
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QPushButton, QFileDialog
from deeplabcut.create_project import add_new_videos
from deeplabcut_gui.dlc_params import DLCParams
from deeplabcut_gui.components import DefaultTab
from deeplabcut_gui.widgets import ConfigEditor


class ManageProject(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super().__init__(root, parent, h1_description)
        self._set_page()
        self._videos = []

    def _set_page(self):
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
