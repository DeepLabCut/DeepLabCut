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
import os
from datetime import datetime

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QDesktopServices, QIcon

import deeplabcut
from deeplabcut.gui import BASE_DIR
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.tabs.docs import (
    URL_3D,
    URL_MA_CONFIGURE,
    URL_USE_GUIDE_SCENARIO,
)
from deeplabcut.gui.widgets import ClickableLabel, ItemSelectionFrame, YesNoSwitch, DynamicTextList
from deeplabcut.utils import auxiliaryfunctions


class ProjectCreator(QtWidgets.QDialog):
    """Project creation dialog"""

    def __init__(self, parent):
        super(ProjectCreator, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("New Project")
        self.setModal(True)
        self.setMinimumWidth(parent.screen_width // 2)
        today = datetime.today().strftime("%Y-%m-%d")
        self.name_default = "-".join(("{}", "{}", today))
        self.proj_default = ""
        self.exp_default = ""
        self.loc_default = parent.project_folder

        self.bodypart_list = None
        self.individuals_list = None
        self.unique_bodyparts_list = None

        self.toggle_3d = YesNoSwitch()
        self.toggle_3d.setChecked(False)
        self.madlc_toggle = YesNoSwitch()
        self.madlc_toggle.setChecked(False)
        self.unique_toggle = YesNoSwitch()
        self.unique_toggle.setChecked(False)
        self.identity_toggle = YesNoSwitch()
        self.identity_toggle.setChecked(False)

        main_layout = QtWidgets.QVBoxLayout(self) # this?
        self.user_frame = self.lay_out_user_frame()
        self.video_frame = self.lay_out_video_frame()
        self.create_button = QtWidgets.QPushButton("Create")
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self.finalize_project)
        main_layout.addWidget(self.user_frame)
        main_layout.addWidget(self.video_frame)
        main_layout.addWidget(self.create_button, alignment=QtCore.Qt.AlignRight)

    def lay_out_user_frame(self):
        user_frame = QtWidgets.QFrame(self)
        user_frame.setFrameShape(user_frame.Shape.StyledPanel)
        user_frame.setLineWidth(0)

        proj_label = QtWidgets.QLabel("Project:", user_frame)
        self.proj_line = QtWidgets.QLineEdit(self.proj_default, user_frame)
        self.proj_line.setPlaceholderText("my project's name")
        self._default_style = self.proj_line.styleSheet()
        self.proj_line.textEdited.connect(self.update_project_name)

        exp_label = QtWidgets.QLabel("Experimenter:", user_frame)
        self.exp_line = QtWidgets.QLineEdit(self.exp_default, user_frame)
        self.exp_line.setPlaceholderText("my nickname")
        self.exp_line.textEdited.connect(self.update_experimenter_name)

        loc_label = ClickableLabel("Location:", parent=user_frame)
        loc_label.signal.connect(self.on_click)
        self.loc_line = QtWidgets.QLineEdit(self.loc_default, user_frame)
        self.loc_line.setReadOnly(True)
        action = self.loc_line.addAction(
            QIcon(os.path.join(BASE_DIR, "assets", "icons", "open2.png")),
            QtWidgets.QLineEdit.TrailingPosition,
        )
        action.triggered.connect(self.on_click)

        vbox = QtWidgets.QVBoxLayout(user_frame)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(proj_label, 0, 0)
        grid.addWidget(self.proj_line, 0, 1)
        grid.addWidget(exp_label, 1, 0)
        grid.addWidget(self.exp_line, 1, 1)
        grid.addWidget(loc_label, 2, 0)
        grid.addWidget(self.loc_line, 2, 1)
        vbox.addLayout(grid)

        widget_3d = self.build_toggle_widget(
            switch=self.toggle_3d,
            question="Do you want to create a 3D pose estimation project?",
            help_text="(What is needed for a 3D project?)",
            docs_link=URL_3D,
        )
        madlc_widget = self.build_toggle_widget(
            switch=self.madlc_toggle,
            question="Are there multiple individuals in your videos?",
            help_text="(Why does this matter?)",
            docs_link=URL_USE_GUIDE_SCENARIO,
        )

        # Only visible when the maDLC widget is checked
        unique_widget = self.build_toggle_widget(
            switch=self.unique_toggle,
            question="Do you have unique bodyparts in your video?",
            help_text="(What are unique bodyparts?)",
            docs_link=URL_MA_CONFIGURE,
        )
        unique_widget.setVisible(False)

        # Labelling with identity
        identity_widget = self.build_toggle_widget(
            switch=self.identity_toggle,
            question="Label with identity?",
            help_text="(What is labeling with identity?)",
            docs_link=URL_MA_CONFIGURE,
        )
        identity_widget.setVisible(False)

        vbox.addWidget(widget_3d, alignment=QtCore.Qt.AlignTop)
        vbox.addWidget(madlc_widget, alignment=QtCore.Qt.AlignTop)
        vbox.addWidget(unique_widget, alignment=QtCore.Qt.AlignTop)
        vbox.addWidget(identity_widget, alignment=QtCore.Qt.AlignTop)

        # Create horizontal layout for the two lists
        lists_layout = QtWidgets.QHBoxLayout()
        lists_layout.setAlignment(QtCore.Qt.AlignTop)

        # Create both DynamicTextList widgets as class attributes
        self.bodypart_list = DynamicTextList(
            label_text="Bodyparts to track",
            parent=self,
        )

        self.individuals_list = DynamicTextList(
            label_text="Individual names",
            parent=self,
        )
        self.individuals_list.setVisible(False)

        self.unique_bodyparts_list = DynamicTextList(
            label_text="Unique bodyparts to track",
            parent=self,
        )
        self.unique_bodyparts_list.setVisible(False)

        # Connect toggle state to individuals list visibility, unique, identity
        self.madlc_toggle.toggled.connect(self.individuals_list.setVisible)
        self.madlc_toggle.toggled.connect(unique_widget.setVisible)
        self.madlc_toggle.toggled.connect(identity_widget.setVisible)

        # Connect the unique_toggle to the unique_bodyparts_list
        self.unique_toggle.toggled.connect(
            lambda yes: self.unique_bodyparts_list.setVisible(
                yes and self.madlc_toggle.isChecked()
            )
        )

        # Connect 3d toggle to all other option visibility
        self.toggle_3d.toggled.connect(lambda yes: madlc_widget.setVisible(not yes))
        self.toggle_3d.toggled.connect(
            lambda checked_3d: unique_widget.setVisible(
                not checked_3d and self.madlc_toggle.isChecked()
            )
        )
        self.toggle_3d.toggled.connect(
            lambda checked_3d: identity_widget.setVisible(
                not checked_3d and self.madlc_toggle.isChecked()
            )
        )
        self.toggle_3d.toggled.connect(
            lambda checked_3d: self.bodypart_list.setVisible(not checked_3d)
        )
        self.toggle_3d.toggled.connect(
            lambda checked_3d: self.individuals_list.setVisible(
                not checked_3d and self.madlc_toggle.isChecked()
            )
        )
        self.toggle_3d.toggled.connect(
            lambda checked_3d: self.unique_bodyparts_list.setVisible(
                not checked_3d
                and self.madlc_toggle.isChecked()
                and self.unique_toggle.isChecked()
            )
        )

        # Add both lists to the horizontal layout with top alignment
        lists_layout.addWidget(self.bodypart_list, alignment=QtCore.Qt.AlignTop)
        lists_layout.addWidget(self.individuals_list, alignment=QtCore.Qt.AlignTop)
        lists_layout.addWidget(self.unique_bodyparts_list, alignment=QtCore.Qt.AlignTop)

        # Add the horizontal layout to the main vertical layout
        vbox.addLayout(lists_layout)
        return user_frame

    def build_toggle_widget(
        self,
        switch: YesNoSwitch,
        question: str,
        help_text: str,
        docs_link: str,
    ) -> QtWidgets.QWidget:
        toggle_layout = QtWidgets.QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(10)

        toggle_label = QtWidgets.QLabel(question)
        toggle_label.setAlignment(QtCore.Qt.AlignLeft)
        help_label = ClickableLabel(help_text, parent=self)
        help_label.setStyleSheet("text-decoration: underline; font-weight: bold;")
        help_label.setCursor(QtCore.Qt.PointingHandCursor)
        help_label.signal.connect(
            lambda: QDesktopServices.openUrl(QtCore.QUrl(docs_link))
        )

        toggle_layout.addWidget(switch, alignment=QtCore.Qt.AlignLeft)
        toggle_layout.addWidget(toggle_label, alignment=QtCore.Qt.AlignLeft)
        toggle_layout.addStretch()
        toggle_layout.addWidget(help_label, alignment=QtCore.Qt.AlignRight)
        toggle_widget = QtWidgets.QWidget()
        toggle_widget.setLayout(toggle_layout)
        return toggle_widget

    def lay_out_video_frame(self):
        video_frame = ItemSelectionFrame([], self)

        self.copy_box = QtWidgets.QCheckBox("Copy videos to project folder")
        self.copy_box.setChecked(False)

        browse_button = QtWidgets.QPushButton("Browse folders for videos")
        browse_button.clicked.connect(self.browse_videos)
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(video_frame.fancy_list.clear)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(browse_button)
        layout.addWidget(clear_button)
        video_frame.layout.addLayout(layout)
        video_frame.layout.addWidget(self.copy_box)

        self.toggle_3d.toggled.connect(lambda yes: self.copy_box.setVisible(not yes))
        self.toggle_3d.toggled.connect(lambda yes: browse_button.setVisible(not yes))
        self.toggle_3d.toggled.connect(lambda yes: clear_button.setVisible(not yes))
        self.toggle_3d.toggled.connect(lambda yes: video_frame.setVisible(not yes))
        return video_frame

    def browse_videos(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Please select a folder",
            self.loc_default,
            options,
        )
        if not folder:
            return

        for video in auxiliaryfunctions.grab_files_in_folder(
            folder,
            relative=False,
        ):
            if os.path.splitext(video)[1][1:].lower() in DLCParams.VIDEOTYPES[1:]:
                self.video_frame.fancy_list.add_item(video)

    def finalize_project(self):
        fields = [self.proj_line, self.exp_line]
        empty = [i for i, field in enumerate(fields) if not field.text()]
        for i, field in enumerate(fields):
            if i in empty:
                field.setStyleSheet("border: 1px solid red;")
            else:
                field.setStyleSheet(self._default_style)
        if empty:
            return

        create_3d = self.toggle_3d.isChecked()
        try:
            if create_3d:
                _ = deeplabcut.create_new_project_3d(
                    self.proj_default,
                    self.exp_default,
                    2,
                    self.loc_default,
                )
            else:
                videos = list(self.video_frame.selected_items)
                if not len(videos):
                    print("Add at least a video to the project.")
                    self.video_frame.fancy_list.setStyleSheet("border: 1px solid red")
                    return
                else:
                    self.video_frame.fancy_list.setStyleSheet(
                        self.video_frame.fancy_list._default_style
                    )
                to_copy = self.copy_box.isChecked()
                is_madlc = self.madlc_toggle.isChecked()
                config = deeplabcut.create_new_project(
                    self.proj_default,
                    self.exp_default,
                    videos,
                    self.loc_default,
                    to_copy,
                    multianimal=is_madlc,
                )

                if self.bodypart_list is not None:
                    bodypart_key = "bodyparts"
                    updates = {}
                    if is_madlc:
                        bodypart_key = "multianimalbodyparts"
                        if self.individuals_list is not None:
                            individuals = self.individuals_list.get_entries()
                            if len(individuals) > 0:
                                updates["individuals"] = individuals

                        if (
                            self.unique_toggle.isChecked()
                            and self.unique_bodyparts_list is not None
                        ):
                            unique_bodyparts = self.unique_bodyparts_list.get_entries()
                            if len(unique_bodyparts) > 0:
                                updates["uniquebodyparts"] = unique_bodyparts

                        if self.identity_toggle.isChecked():
                            updates["identity"] = True

                    bodyparts = self.bodypart_list.get_entries()
                    if len(bodyparts) > 0:
                        updates[bodypart_key] = bodyparts

                    if len(updates) > 0:
                        cfg: dict = auxiliaryfunctions.read_config(config)
                        cfg.update(**updates)
                        auxiliaryfunctions.write_config(config, cfg)

                self.parent.load_config(config)
                self.parent._update_project_state(config=config, loaded=True)
        except FileExistsError:
            print('Project "{}" already exists!'.format(self.proj_default))
            return

        msg = QtWidgets.QMessageBox(text="New project created")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()

        self.close()

    def on_click(self):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Please select a folder", self.loc_default
        )
        if not dirname:
            return
        self.loc_default = dirname
        self.update_project_location()

    def update_project_name(self, text):
        self.proj_default = text
        self.update_project_location()

    def update_experimenter_name(self, text):
        self.exp_default = text
        self.update_project_location()

    def update_project_location(self):
        full_name = self.name_default.format(self.proj_default, self.exp_default)
        full_path = os.path.join(self.loc_default, full_name)
        self.loc_line.setText(full_path)
