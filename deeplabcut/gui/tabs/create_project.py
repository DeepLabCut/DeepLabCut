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
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPainter, QPen

import deeplabcut
from deeplabcut.gui import BASE_DIR
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.widgets import ClickableLabel, ItemSelectionFrame
from deeplabcut.gui.tabs.docs import (
    URL_3D,
    URL_MA_CONFIGURE,
    URL_USE_GUIDE_SCENARIO,
)
from deeplabcut.utils import auxiliaryfunctions


class DynamicTextList(QtWidgets.QWidget):
    """Dynamically add text entries"""

    def __init__(self, label_text="bodyparts", parent=None):
        super(DynamicTextList, self).__init__(parent)
        self.label_text = label_text
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Set maximum width for the widget
        self.setMaximumWidth(300)

        # Add explanatory label
        label = QtWidgets.QLabel(label_text)
        self.layout.addWidget(label)

        # Create scroll area and its widget
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)  # Remove frame border

        # Create widget to hold the entries
        self.entries_widget = QtWidgets.QWidget()
        self.entries_layout = QtWidgets.QVBoxLayout(self.entries_widget)
        self.entries_layout.setContentsMargins(0, 0, 0, 0)
        self.entries_layout.setSpacing(5)  # Consistent spacing between entries
        self.entries_layout.setAlignment(QtCore.Qt.AlignTop)  # Align entries to top

        # Add stretch at the bottom to keep entries at top
        self.entries_layout.addStretch()

        self.scroll.setWidget(self.entries_widget)

        # Set fixed height for 6 items
        self.entry_height = 30  # Fixed height for each entry
        self.padding = 10  # Extra padding
        self.scroll.setFixedHeight(5 * self.entry_height + self.padding)

        # Add scroll area to main layout
        self.layout.addWidget(self.scroll)

        self.entries = []
        self.add_entry()

    def add_entry(self):
        # Create horizontal layout for index and entry
        entry_layout = QtWidgets.QHBoxLayout()
        entry_layout.setContentsMargins(0, 0, 10, 0)
        entry_layout.setSpacing(5)  # Consistent spacing between index and entry

        # Create container widget for the entry row
        entry_widget = QtWidgets.QWidget()
        entry_widget.setFixedHeight(self.entry_height)
        entry_widget.setLayout(entry_layout)

        # Add index label
        index_label = QtWidgets.QLabel(str(len(self.entries) + 1) + ".")
        index_label.setFixedWidth(20)  # Set fixed width for alignment
        entry_layout.addWidget(index_label)

        # Add text entry
        entry = QtWidgets.QLineEdit()
        entry.setFixedHeight(self.entry_height - 6)  # Slightly smaller than container
        entry.textChanged.connect(self._on_text_changed)
        entry.textEdited.connect(lambda text: self._check_for_spaces(entry, text))
        self.entries.append((entry, index_label))  # Store both widgets
        entry_layout.addWidget(entry)

        # Insert the new entry before the stretch
        self.entries_layout.insertWidget(len(self.entries) - 1, entry_widget)

    def _check_for_spaces(self, entry, text):
        if " " in text:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText(
                f"Spaces are not allowed in the {self.label_text} list. Use underscores "
                f"instead."
            )
            msg.setWindowTitle("Warning")
            msg.exec_()
            entry.setText(entry.text().replace(" ", "_"))

    def _on_text_changed(self):
        # If the last entry has text, add a new empty entry
        if self.entries[-1][0].text():
            self.add_entry()

        # Remove any empty entries except the last one
        entries_to_remove = []
        for i, (entry, _) in enumerate(self.entries[:-1]):
            if not entry.text():
                entries_to_remove.append(i)

        for i in reversed(entries_to_remove):
            entry_widget = self.entries[i][0].parent()
            self.entries_layout.removeWidget(entry_widget)
            entry_widget.deleteLater()
            self.entries.pop(i)

        self._update_indices()  # Update the indices after removal

    def get_entries(self):
        return [entry[0].text() for entry in self.entries if entry[0].text()]

    def _update_indices(self):
        for i, (entry, index_label) in enumerate(self.entries):
            index_label.setText(str(i + 1) + ".")


class Switch(QtWidgets.QPushButton):

    def __init__(self, on_text="Yes", off_text="No", width=80, parent=None):
        super().__init__(parent)
        self.on_text = on_text
        self.off_text = off_text
        self.setCheckable(True)
        self.setFixedWidth(width)
        self.setMinimumHeight(22)

    def paintEvent(self, event):
        # Colors: https://qdarkstylesheet.readthedocs.io/en/latest/color_reference.html
        label = self.on_text if self.isChecked() else self.off_text
        bg_color = "#00ff00" if self.isChecked() else "#9DA9B5"

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QColor(69, 83, 100))  # Lighter gray background

        pen = QPen("#455364")
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(
            QtCore.QRect(-width, -radius, 2 * width, 2 * radius), radius, radius
        )
        painter.setBrush(QBrush(bg_color))
        sw_rect = QtCore.QRect(-radius, -radius, width + radius, 2 * radius)
        if not self.isChecked():
            sw_rect.moveLeft(-width)

        painter.drawRoundedRect(sw_rect, radius, radius)

        pen = QPen("#000000")
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawText(sw_rect, QtCore.Qt.AlignCenter, label)


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

        self.toggle_3d = Switch()
        self.toggle_3d.setChecked(False)
        self.madlc_toggle = Switch()
        self.madlc_toggle.setChecked(False)
        self.unique_toggle = Switch()
        self.unique_toggle.setChecked(False)
        self.identity_toggle = Switch()
        self.identity_toggle.setChecked(False)

        main_layout = QtWidgets.QVBoxLayout(self)
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
        switch: Switch,
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
