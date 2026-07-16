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
"""Tests for deeplabcut/pose_estimation_pytorch/apis/utils.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.apis.utils import return_train_network_path

_READ_CONFIG = "deeplabcut.pose_estimation_pytorch.apis.utils.auxiliaryfunctions.read_config"
_GET_MODEL_FOLDER = "deeplabcut.pose_estimation_pytorch.apis.utils.auxiliaryfunctions.get_model_folder"

_PROJECT_PATH = Path("/project")
_MODEL_FOLDER = "dlc-models/iteration-0/trainset80shuffle1"

# Mock ProjectConfig with minimal attribute + dict access
_DATA = {"project_path": Path("/project"), "TrainingFraction": [0.8, 0.95]}
_CFG = MagicMock(**_DATA)
_CFG.__getitem__.side_effect = _DATA.__getitem__


# ---------------------------------------------------------------------------
# return_train_network_path
# ---------------------------------------------------------------------------


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_returns_three_paths(mock_read_config, mock_get_model_folder):
    result = return_train_network_path("cfg.yaml")
    assert len(result) == 3
    assert all(isinstance(p, Path) for p in result)


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_reads_config(mock_read_config, mock_get_model_folder):
    return_train_network_path("cfg.yaml")
    mock_read_config.assert_called_once_with("cfg.yaml")


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_passes_correct_args_to_get_model_folder(mock_read_config, mock_get_model_folder):
    return_train_network_path("cfg.yaml", shuffle=3, trainingsetindex=1, modelprefix="pfx")
    mock_get_model_folder.assert_called_once_with(
        0.95,  # TrainingFraction[1]
        3,
        _CFG,
        engine=Engine.PYTORCH,
        modelprefix="pfx",
    )


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_train_config_path(mock_read_config, mock_get_model_folder):
    train_cfg, _, _ = return_train_network_path("cfg.yaml")
    expected = Path(_PROJECT_PATH) / _MODEL_FOLDER / "train" / "pytorch_config.yaml"
    assert train_cfg == expected


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_test_config_path(mock_read_config, mock_get_model_folder):
    _, test_cfg, _ = return_train_network_path("cfg.yaml")
    expected = Path(_PROJECT_PATH) / _MODEL_FOLDER / "test" / "pose_cfg.yaml"
    assert test_cfg == expected


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_snapshot_folder(mock_read_config, mock_get_model_folder):
    _, _, snapshot_folder = return_train_network_path("cfg.yaml")
    expected = Path(_PROJECT_PATH) / _MODEL_FOLDER / "train"
    assert snapshot_folder == expected


@patch(_GET_MODEL_FOLDER, return_value=_MODEL_FOLDER)
@patch(_READ_CONFIG, return_value=_CFG)
def test_return_train_network_path_default_uses_shuffle_1_and_first_training_fraction(
    mock_read_config, mock_get_model_folder
):
    return_train_network_path("cfg.yaml")
    mock_get_model_folder.assert_called_once_with(
        0.8,  # TrainingFraction[0]
        1,  # default shuffle=1
        _CFG,
        engine=Engine.PYTORCH,
        modelprefix="",
    )
