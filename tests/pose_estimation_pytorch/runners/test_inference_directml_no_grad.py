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
"""Tests DLC_DIRECTML_NO_GRAD toggles inference_mode vs no_grad (AMD DirectML)."""

from __future__ import annotations

import importlib
import os
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners.inference as inference


def _reload_with_env(env_value: str | None):
    if env_value is None:
        os.environ.pop("DLC_DIRECTML_NO_GRAD", None)
    else:
        os.environ["DLC_DIRECTML_NO_GRAD"] = env_value
    importlib.reload(inference)


@pytest.fixture(autouse=True)
def _restore_env():
    yield
    _reload_with_env(None)  # always restore defaults after each test


@pytest.mark.parametrize(
    ("env_value", "directml_no_grad"),
    [(None, False), ("false", False), ("true", True)],
)
def test_directml_no_grad_env(env_value, directml_no_grad):
    """env var sets _directml_no_grad and selects the correct torch grad context."""
    _reload_with_env(env_value)
    assert inference._directml_no_grad is directml_no_grad

    class _SniffRunner(inference.InferenceRunner):
        def __init__(self):
            super().__init__(
                model=Mock(),
                batch_size=1,
                inference_cfg=inference.InferenceConfig(
                    multithreading=inference.MultithreadingConfig(enabled=False),
                ),
            )
            self.saw_inference_mode: bool | None = None

        def predict(self, inputs: torch.Tensor, **kwargs):
            self.saw_inference_mode = torch.is_inference_mode_enabled()
            return [{"mock": {"poses": np.zeros((1,), dtype=np.float32)}}]

    runner = _SniffRunner()
    runner.inference([np.zeros((1, 3, 8, 8), dtype=np.float32)])

    assert runner.saw_inference_mode is not directml_no_grad
