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
"""Tests for the training API."""

from pathlib import Path
from unittest.mock import Mock, patch

from deeplabcut.pose_estimation_pytorch.apis.training import train
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from deeplabcut.pose_estimation_pytorch.task import Task


def _minimal_run_config(tmp_path: Path, *, resume_from: str | None = None) -> PoseConfig:
    project_cfg = {
        "multianimalproject": False,
        "project_path": str(tmp_path),
        "bodyparts": ["nose"],
        "uniquebodyparts": [],
        "individuals": ["mouse"],
    }
    cfg_path = tmp_path / "pytorch_config.yaml"
    pose_config = make_pytorch_pose_config(project_cfg, str(cfg_path), net_type="resnet_50")
    if resume_from is not None:
        pose_config.resume_training_from = resume_from
    return pose_config


@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.PoseModel.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_uses_resume_training_from_config(
    mock_build_runner: Mock,
    mock_build_model: Mock,
    mock_build_transforms: Mock,
    tmp_path: Path,
) -> None:
    run_config = _minimal_run_config(tmp_path, resume_from="/train/snapshot-010.pt")

    loader = Mock()
    loader.model_folder = tmp_path
    loader.model_cfg = run_config
    train_dataset = Mock(__len__=Mock(return_value=1))
    valid_dataset = Mock(__len__=Mock(return_value=1))
    loader.create_dataset = Mock(side_effect=[train_dataset, valid_dataset])

    train(loader=loader, run_config=run_config, task=Task.BOTTOM_UP, device="cpu", snapshot_path=None)

    assert mock_build_runner.call_args.kwargs["snapshot_path"] == "/train/snapshot-010.pt"
