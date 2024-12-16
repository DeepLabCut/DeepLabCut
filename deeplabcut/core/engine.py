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
"""Defines the deep learning frameworks available"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class EngineDataMixin:
    aliases: tuple[str]
    model_folder_name: str
    pose_cfg_name: str
    results_folder_name: str


class Engine(EngineDataMixin, Enum):
    PYTORCH = (
        ("pytorch", "torch"),
        "dlc-models-pytorch",
        "pytorch_config.yaml",
        "evaluation-results-pytorch",
    )
    TF = (
        ("tensorflow", "tf"),
        "dlc-models",
        "pose_cfg.yaml",
        "evaluation-results",
    )

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if value.lower() in member.aliases:
                    return member
        return None

    def __repr__(self) -> str:
        return f"Engine.{self.name}"
