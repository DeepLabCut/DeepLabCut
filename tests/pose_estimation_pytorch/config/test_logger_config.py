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
"""Tests for WandbLoggerConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeplabcut.pose_estimation_pytorch.config.logger import LoggerType, WandbLoggerConfig


def test_wandb_kwargs_accepts_nested_extras() -> None:
    cfg = WandbLoggerConfig.from_dict(
        {
            "type": LoggerType.WandbLogger,
            "wandb_kwargs": {
                "entity": "team",
                "mode": "offline",
                "tags": ["a"],
                "settings": {"_service_wait": 60},
            },
        }
    )
    assert cfg.wandb_kwargs["entity"] == "team"
    assert cfg.wandb_kwargs["settings"] == {"_service_wait": 60}


def test_scoops_top_level_extras_into_wandb_kwargs() -> None:
    cfg = WandbLoggerConfig.from_dict(
        {
            "type": LoggerType.WandbLogger,
            "project_name": "proj",
            "group": "group_a",
            "tags": ["mytag"],
        }
    )
    assert cfg.project_name == "proj"
    assert cfg.wandb_kwargs == {"group": "group_a", "tags": ["mytag"]}


def test_mixed_top_level_and_nested_wandb_kwargs() -> None:
    cfg = WandbLoggerConfig.from_dict(
        {
            "type": LoggerType.WandbLogger,
            "group": "group_a",
            "wandb_kwargs": {"tags": ["mytag"], "mode": "offline", "id": "run-1"},
        }
    )
    assert cfg.wandb_kwargs == {
        "tags": ["mytag"],
        "mode": "offline",
        "id": "run-1",
        "group": "group_a",
    }


def test_rejects_duplicate_top_level_and_nested() -> None:
    with pytest.raises(ValidationError, match="Duplicate wandb.init options"):
        WandbLoggerConfig.from_dict(
            {
                "type": LoggerType.WandbLogger,
                "tags": ["a"],
                "wandb_kwargs": {"tags": ["b"], "mode": "offline"},
            }
        )


@pytest.mark.parametrize("overlap_key", ["project", "name"])
def test_wandb_kwargs_rejects_reserved_keys(overlap_key: str) -> None:
    with pytest.raises(ValidationError, match="reserved options"):
        WandbLoggerConfig.from_dict(
            {
                "type": LoggerType.WandbLogger,
                "wandb_kwargs": {overlap_key: "x", "mode": "offline"},
            }
        )
