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
"""Module to display information about the selected shuffle in the GUI"""
from __future__ import annotations
from pathlib import Path

import PySide6.QtCore as QtCore
from PySide6 import QtWidgets

from deeplabcut.core.engine import Engine
from deeplabcut.utils import auxiliaryfunctions


class SelectedShuffleDisplay(QtWidgets.QWidget):
    """A widget displaying information about the selected shuffle"""
    pose_cfg_signal = QtCore.Signal(dict)

    def __init__(self, root, row_margin: int = 25):
        super().__init__()
        self.root = root

        self._row_margin = row_margin

        self._current_index: int | None = None
        self._engine: Engine | None = None
        self._is_top_down: bool = False
        self._net_type: str | None = None
        self._pose_cfg: dict | None = None

        self._label = QtWidgets.QLabel("Shuffle info:")
        self._label.setStyleSheet(f"margin: 0px 0px {self._row_margin}px 0px")
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        self.setLayout(layout)

        # initialize the display
        self._update_display(self.root.shuffle_value)

        # update the display when the shuffle or selected engine changes, or when a new
        # shuffle has been created
        self.root.shuffle_change.connect(self._update_display)
        self.root.engine_change.connect(self._update_display)
        self.root.shuffle_created.connect(self._update_display)

    @property
    def pose_cfg(self) -> dict | None:
        return self._pose_cfg

    @pose_cfg.setter
    def pose_cfg(self, value: dict | None) -> None:
        self._pose_cfg = value
        self.pose_cfg_signal.emit(self._pose_cfg)

    @QtCore.Slot(int)
    def _update_display(self, new_index: int) -> None:
        self._current_index = new_index

        try:
            pose_cfg_path = Path(self.root.pose_cfg_path)
        except ValueError as err:
            self._set_text_error(
                f"Failed to read shuffle {self._current_index} - check that it exists!"
            )
            return
        except ModuleNotFoundError as err:
            # Loading a TF shuffle but TF is not installed
            self._set_text_error(
                f"Failed to read shuffle {self._current_index} due to error `{err}`.\n"
                "If the error is `ModuleNotFoundError: No module named 'tensorflow'`, "
                f"this is because\nshuffle {self._current_index} uses the tensorflow "
                " engine, but TensorFlow is not installed in your environment.\n"
                "Ignore this error if you'll just train PyTorch models. To train "
                "TensorFlow models, install it with \n"
                "    Windows/Linux: pip install 'deeplabcut[tf]'\n"
                "    Apple Silicon: pip install 'deeplabcut[apple_mchips]'"
            )
            return

        if not pose_cfg_path.exists():
            self._set_text_error()
            return

        self._read_pose_config(pose_cfg_path)
        self._set_text()

    def _set_text(self) -> None:
        engine_str = "None"
        if self._engine is not None:
            engine_str = self._engine.aliases[0]

        text = f"net type: {self._net_type}  |  engine: {engine_str}"
        if self._engine == Engine.PYTORCH and self._is_top_down:
            text += f"  |  top-down"

        style = f"margin: 0px 0px {self._row_margin}px 0px;"
        if self._engine != self.root.engine:
            warning = "Change the selected Engine in the top-right to use this shuffle!"
            text = warning + "  |  " + text
            style += " color: orange;"

        self._label.setStyleSheet(style)
        self._label.setText(text)

    def _set_text_error(self, error: str) -> None:
        self._label.setText(error)
        style = f"margin: 0px 0px {self._row_margin}px 0px; color: orange;"
        self._label.setStyleSheet(style)
        self.pose_cfg = None

    def _read_pose_config(self, pose_cfg_path: Path) -> None:
        pose_cfg = auxiliaryfunctions.read_plainconfig(str(pose_cfg_path))

        self._engine = (
            Engine.PYTORCH if "pytorch" in pose_cfg_path.stem.lower() else Engine.TF
        )
        self._net_type = pose_cfg.get("net_type", "UNKNOWN")
        self._is_top_down = (
            self._engine == Engine.PYTORCH and pose_cfg.get("method").lower() == "td"
        )
        self.pose_cfg = pose_cfg
