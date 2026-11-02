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
"""Tests for RunnerConfig validation."""

from __future__ import annotations

import pytest

from deeplabcut.pose_estimation_pytorch.config.runner import RunnerConfig


@pytest.mark.parametrize("value", [True, False])
def test_accepts_load_scheduler_state_dict(value: bool) -> None:
    """Regression test for #3556: the documented key was rejected as extra."""
    cfg = RunnerConfig.from_dict({"type": "PoseTrainingRunner", "load_scheduler_state_dict": value})
    assert cfg.load_scheduler_state_dict is value
    assert cfg.to_dict()["load_scheduler_state_dict"] is value


def test_load_scheduler_state_dict_defaults_to_true() -> None:
    """The default matches the fallback used when building the training runner."""
    cfg = RunnerConfig.from_dict({"type": "PoseTrainingRunner"})
    assert cfg.load_scheduler_state_dict is True
