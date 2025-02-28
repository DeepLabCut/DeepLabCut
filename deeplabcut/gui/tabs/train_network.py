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
from dataclasses import dataclass

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon

import deeplabcut.compat as compat
from deeplabcut.core.engine import Engine
from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    SnapshotSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
)
from deeplabcut.gui.displays.selected_shuffle_display import SelectedShuffleDisplay
from deeplabcut.gui.widgets import ConfigEditor


@dataclass
class IntTrainAttribute:
    label: str
    fn_key: str
    default: int
    min: int
    max: int
    tooltip: str | None = None


@dataclass
class TrainAttributeRow:
    attributes: list[IntTrainAttribute]
    description: str | None = None
    show_when_cfg: tuple[str, str] | None = None


class TrainNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)
        self._shuffle: ShuffleSpinBox = ShuffleSpinBox(root=self.root, parent=self)
        self._shuffle_display = SelectedShuffleDisplay(self.root)

        self._attribute_layouts: dict[Engine, QtWidgets.QWidget] = {}
        self._attribute_kwargs: dict[Engine, dict] = {}
        self._rows_with_requirements: list = []
        self._set_page()

        self.root.engine_change.connect(self._on_engine_change)
        self._shuffle_display.pose_cfg_signal.connect(self._pose_cfg_change)

    @Slot(Engine)
    def _on_engine_change(self, engine: Engine) -> None:
        for e, layout in self._attribute_layouts.items():
            if e == engine:
                layout.show()
            else:
                layout.hide()
        self._update_snapshot_selection_widgets_visibility()

    def _update_snapshot_selection_widgets_visibility(self):
        if self.root.engine == Engine.PYTORCH:
            self.resume_from_snapshot_label.show()
            self.snapshot_selection_widget.show()
            # Display detector snapshot selection widget only if in Top-Down mode
            if self._shuffle_display.pose_cfg.get("method", "").lower() == "td":
                self.detector_snapshot_selection_widget.show()
            else:
                self.detector_snapshot_selection_widget.hide()
        else:
            self.resume_from_snapshot_label.hide()
            self.snapshot_selection_widget.hide()
            self.detector_snapshot_selection_widget.hide()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self._generate_layout_attributes()

        self.resume_from_snapshot_label = _create_label_widget(
            "[Optional]: Select a snapshot to resume training from", "font:bold"
        )
        self.resume_from_snapshot_label.setToolTip(
            "<span style='font-weight:normal; white-space:nowrap;'>"
            "If you've already trained a model on this shuffle, you can continue training it instead of starting "
            "from scratch again. <br>When using top-down models, you can also choose a detector to resume training from."
            "</span>"
        )
        self.main_layout.addWidget(self.resume_from_snapshot_label)

        self.snapshot_selection_widget = SnapshotSelectionWidget(
            self.root, self, margins=(30, 0, 0, 0), select_button_text="Select snapshot"
        )
        self.main_layout.addWidget(self.snapshot_selection_widget)

        self.detector_snapshot_selection_widget = SnapshotSelectionWidget(
            self.root,
            self,
            margins=(30, 0, 0, 0),
            select_button_text="Select detector snapshot",
        )
        self.main_layout.addWidget(self.detector_snapshot_selection_widget)

        self._pose_cfg_change(
            self._shuffle_display.pose_cfg
        )  # also calls _update_snapshot_selection_widgets_visibility

        self.edit_posecfg_btn = QtWidgets.QPushButton("Edit pose_cfg.yaml")
        self.edit_posecfg_btn.setMinimumWidth(150)
        self.edit_posecfg_btn.clicked.connect(self.open_posecfg_editor)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.train_network)

        self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(compat.train_network.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self) -> None:
        row_margin = 25

        # top layout
        shuffle_label = QtWidgets.QLabel("Shuffle")
        shuffle_label.setStyleSheet(f"margin: 0px 0px {row_margin}px 0px")
        self._shuffle.setStyleSheet(f"margin: 0px 0px {row_margin}px 0px")
        self._shuffle_display.setStyleSheet(f"margin: 0px 0px {row_margin}px 0px")

        base_layout = _create_grid_layout(margins=(20, 0, 0, 0))
        base_layout.addWidget(shuffle_label, 0, 0)
        base_layout.addWidget(self._shuffle, 0, 1)
        base_layout.addWidget(self._shuffle_display, 0, 2)
        base_layout_widget = QtWidgets.QWidget()
        base_layout_widget.setLayout(base_layout)
        self.main_layout.addWidget(base_layout_widget)

        for engine in Engine:
            train_attributes = get_train_attributes(engine)

            # Other parameters
            param_layout = _create_grid_layout(margins=(20, 0, 0, 0))
            param_layout.setVerticalSpacing(0)

            self._attribute_kwargs[engine] = {}
            row_index = 1
            for row in train_attributes:
                row_elements = []
                if row.description is not None:
                    row_label = QtWidgets.QLabel(row.description)
                    row_label.setStyleSheet("font-weight: bold")
                    row_elements.append(row_label)
                    param_layout.addWidget(row_label, row_index, 0)
                    row_index += 1

                for j, attribute in enumerate(row.attributes):
                    label = QtWidgets.QLabel(attribute.label)
                    spin_box = QtWidgets.QSpinBox()
                    spin_box.setMinimum(attribute.min)
                    spin_box.setMaximum(attribute.max)
                    spin_box.setValue(attribute.default)
                    spin_box.valueChanged.connect(
                        lambda new_val: self.log_attribute_change(attribute, new_val)
                    )
                    self._attribute_kwargs[engine][attribute.fn_key] = spin_box

                    # Pad below to create spacing with other rows
                    label.setStyleSheet(f"margin: 0px 0px {row_margin}px 0px")
                    spin_box.setStyleSheet(f"margin: 0px 0px {row_margin}px 0px")

                    row_elements.append(label)
                    row_elements.append(spin_box)

                    param_layout.addWidget(label, row_index, 2 * j)
                    param_layout.addWidget(spin_box, row_index, 2 * j + 1)

                if row.show_when_cfg is not None:
                    self._rows_with_requirements.append(
                        (row.show_when_cfg, row_elements)
                    )

                row_index += 1

            layout_widget = QtWidgets.QWidget()
            layout_widget.setLayout(param_layout)
            self._attribute_layouts[engine] = layout_widget
            if engine != self.root.engine:
                layout_widget.hide()

            self.main_layout.addWidget(layout_widget)

    def log_attribute_change(self, attribute: IntTrainAttribute, value: int) -> None:
        self.root.logger.info(f"{attribute.label} set to {value}")

    def open_posecfg_editor(self):
        editor = ConfigEditor(self.root.pose_cfg_path)
        editor.show()

    def train_network(self):
        config = self.root.config
        shuffle = int(self._shuffle.value())

        kwargs = dict(gputouse=None, autotune=False)
        for k, spin_box in self._attribute_kwargs[self.root.engine].items():
            kwargs[k] = int(spin_box.value())
        if self.root.engine == Engine.PYTORCH:
            snapshot_to_start_training_from = (
                self.snapshot_selection_widget.selected_snapshot
            )
            if snapshot_to_start_training_from is not None:
                kwargs["snapshot_path"] = snapshot_to_start_training_from
            detector_to_start_training_from = (
                self.detector_snapshot_selection_widget.selected_snapshot
            )
            if detector_to_start_training_from is not None:
                kwargs["detector_path"] = detector_to_start_training_from

        compat.train_network(config, shuffle, **kwargs)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The network is now trained and ready to evaluate.")
        msg.setInformativeText(
            "Use the function 'evaluate_network' to evaluate the network."
        )

        msg.setWindowTitle("Info")
        msg.setMinimumWidth(900)
        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    @Slot(dict)
    def _pose_cfg_change(self, pose_cfg: dict | None) -> None:
        if pose_cfg is None:
            return

        for requirement, widgets in self._rows_with_requirements:
            key, value = requirement
            show = pose_cfg.get(key) == value
            for w in widgets:
                if show:
                    w.show()
                else:
                    w.hide()

        self._update_snapshot_selection_widgets_visibility()


def get_train_attributes(engine: Engine) -> list[TrainAttributeRow]:
    if engine == Engine.TF:
        return [
            TrainAttributeRow(
                attributes=[
                    IntTrainAttribute(
                        label="Display iterations",
                        fn_key="displayiters",
                        default=1000,
                        min=1,
                        max=1000,
                    ),
                    IntTrainAttribute(
                        label="Number of snapshots to keep",
                        fn_key="max_snapshots_to_keep",
                        default=5,
                        min=1,
                        max=100,
                    ),
                ],
            ),
            TrainAttributeRow(
                attributes=[
                    IntTrainAttribute(
                        label="Maximum iterations",
                        fn_key="maxiters",
                        default=100_000,
                        min=1,
                        max=1_030_000,
                    ),
                    IntTrainAttribute(
                        label="Save iterations",
                        fn_key="saveiters",
                        default=50_000,
                        min=1,
                        max=50_000,
                    ),
                ],
            ),
        ]
    elif engine == Engine.PYTORCH:
        return [
            TrainAttributeRow(
                attributes=[
                    IntTrainAttribute(
                        label="Display iterations",
                        fn_key="displayiters",
                        default=1_000,
                        min=1,
                        max=100_000,
                    ),
                    IntTrainAttribute(
                        label="Number of snapshots to keep",
                        fn_key="max_snapshots_to_keep",
                        default=5,
                        min=1,
                        max=100,
                    ),
                ],
            ),
            TrainAttributeRow(
                attributes=[
                    IntTrainAttribute(
                        label="Maximum epochs",
                        fn_key="epochs",
                        default=200,
                        min=1,
                        max=1000,
                    ),
                    IntTrainAttribute(
                        label="Save epochs",
                        fn_key="save_epochs",
                        default=50,
                        min=1,
                        max=250,
                    ),
                ],
            ),
            TrainAttributeRow(
                description="Detector parameters",
                show_when_cfg=("method", "td"),
                attributes=[
                    IntTrainAttribute(
                        label="Detector max epochs",
                        fn_key="detector_epochs",
                        default=200,
                        min=0,
                        max=1000,
                        tooltip="",
                    ),
                    IntTrainAttribute(
                        label="Detector save epochs",
                        fn_key="detector_save_epochs",
                        default=50,
                        min=1,
                        max=250,
                        tooltip="",
                    ),
                ],
            ),
        ]

    raise NotImplementedError(f"Unknown engine: {engine}")
