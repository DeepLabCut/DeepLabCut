#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for DataLoader throughput settings in TrainSettingsConfig (issue #3547)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deeplabcut.pose_estimation_pytorch.config.training import TrainSettingsConfig


def test_dataloader_throughput_defaults_match_pytorch():
    """New settings default to current behaviour (no persistent workers)."""
    cfg = TrainSettingsConfig()

    assert cfg.dataloader_persistent_workers is False
    assert cfg.dataloader_prefetch_factor is None


def test_dataloader_throughput_settings_roundtrip():
    """Settings can be set via config updates and persist to dict."""
    cfg = TrainSettingsConfig(
        dataloader_workers=4,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
    )

    assert cfg.dataloader_persistent_workers is True
    assert cfg.dataloader_prefetch_factor == 4
    as_dict = cfg.to_dict(normalize=True)
    assert as_dict["dataloader_persistent_workers"] is True
    assert as_dict["dataloader_prefetch_factor"] == 4


def test_dataloader_prefetch_factor_must_be_positive():
    """prefetch_factor=0 is rejected at config load, not deep in training."""
    with pytest.raises(ValidationError):
        TrainSettingsConfig(dataloader_prefetch_factor=0)


def test_old_configs_without_throughput_settings_still_validate():
    """Configs written before the new keys existed load with safe defaults.

    This pins the backward-compatibility contract that train() relies on
    when it reads the settings with .get() fallbacks.
    """
    cfg = TrainSettingsConfig.model_validate({"batch_size": 4})

    assert cfg.batch_size == 4
    assert cfg.dataloader_persistent_workers is False
    assert cfg.dataloader_prefetch_factor is None
