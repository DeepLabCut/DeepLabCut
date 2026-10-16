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
"""Tests for TrainSettingsConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeplabcut.pose_estimation_pytorch.config.training import TrainSettingsConfig


def test_dataloader_kwargs_defaults_to_none() -> None:
    assert TrainSettingsConfig().dataloader_kwargs is None


def test_old_configs_without_dataloader_kwargs_still_validate() -> None:
    cfg = TrainSettingsConfig.from_dict({"batch_size": 4, "dataloader_workers": 2})
    assert cfg.dataloader_kwargs is None


def test_dataloader_kwargs_accepts_extra_options() -> None:
    kwargs = {"persistent_workers": True, "prefetch_factor": 4}
    cfg = TrainSettingsConfig.from_dict({"dataloader_workers": 2, "dataloader_kwargs": kwargs})
    assert cfg.dataloader_kwargs == kwargs
    assert cfg.to_dict(normalize=True)["dataloader_kwargs"] == kwargs


@pytest.mark.parametrize(
    ("key", "hint"),
    [
        ("num_workers", "dataloader_workers"),
        ("pin_memory", "dataloader_pin_memory"),
        ("batch_size", "batch_size"),
        ("collate_fn", "data.collate"),
    ],
)
def test_dataloader_kwargs_rejects_options_with_a_config_field(key: str, hint: str) -> None:
    with pytest.raises(ValidationError, match=f"'{key}' \\(set via `{hint}`\\)"):
        TrainSettingsConfig.from_dict({"dataloader_kwargs": {key: 1}})


@pytest.mark.parametrize("key", ["dataset", "shuffle"])
def test_dataloader_kwargs_rejects_options_managed_by_deeplabcut(key: str) -> None:
    with pytest.raises(ValidationError, match=f"dataloader_kwargs cannot include reserved options: '{key}'"):
        TrainSettingsConfig.from_dict({"dataloader_kwargs": {key: False}})
