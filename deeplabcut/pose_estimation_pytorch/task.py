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
"""Types of tasks that can be run by DeepLabCut pose estimation models"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TaskDataMixin:
    aliases: tuple[str]
    snapshot_prefix: str


class Task(TaskDataMixin, Enum):
    """A task to solve"""

    BOTTOM_UP = ("BU", "BottomUp"), "snapshot"
    DETECT = ("DT", "Detect"), "snapshot-detector"
    TOP_DOWN = ("TD", "TopDown"), "snapshot"
    COND_TOP_DOWN = ("CTD", "CondTopDown", "ConditionalTopDown"), "snapshot"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            for member in cls:
                if value in member.aliases:
                    return member
        return None

    def __repr__(self) -> str:
        return f"Task.{self.name}"
