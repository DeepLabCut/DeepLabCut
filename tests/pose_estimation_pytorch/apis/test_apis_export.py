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
"""Tests exporting models"""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.apis.export as export
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.pose_estimation_pytorch import Task
from deeplabcut.pose_estimation_pytorch.runners.snapshots import Snapshot


def _mock_multianimal_project(project_dir: Path):
    video_dir = project_dir / "videos"
    video_dir.mkdir()

    cfg_file, yaml_file = af.create_config_template(multianimal=True)
    cfg_file["Task"] = "mock"
    cfg_file["scorer"] = "mock"
    cfg_file["video_sets"] = {str(video_dir / "vid.mp4"): dict(crop="0, 640, 0, 480")}
    cfg_file["project_path"] = str(project_dir)
    cfg_file["individuals"] = ["a", "b"]
    cfg_file["uniquebodyparts"] = []
    cfg_file["multianimalbodyparts"] = ["k1", "k2", "k3"]
    cfg_file["bodyparts"] = "MULTI!"

    with open(project_dir / "config.yaml", "w") as f:
        yaml_file.dump(cfg_file, f)


def _make_mock_loader(
    project_path: Path,
    project_task: str,
    project_iteration: int,
    model_folder: Path,
    net_type: str,
    pose_task: Task,
    default_snapshot_index: int | str,
    default_detector_snapshot_index: int | str,
) -> Mock:
    loader = Mock()
    loader.project_path = project_path
    loader.model_folder = model_folder
    loader.pose_task = pose_task
    loader.shuffle = 0

    loader.project_cfg = dict(
        task=project_task,
        snapshotindex=default_snapshot_index,
        detector_snapshotindex=default_detector_snapshot_index,
        iteration=project_iteration,
    )
    loader.model_cfg = dict(
        net_type=net_type,
        metadata=dict(
            project_path=str(project_path),
            pose_config_path=str(loader.model_folder / "pytorch_config.yaml"),
        ),
        weight_init=None,
        resume_training_from=None,
    )
    if pose_task == Task.TOP_DOWN:
        loader.model_cfg["detector"] = dict(resume_training_from=None)

    return loader


def _get_export_model_data(project_dir, num_snapshots, task):
    _mock_multianimal_project(project_dir)

    model_dir = Path(project_dir) / "fake-shuffle-0"
    model_dir.mkdir(exist_ok=True)
    snapshots = []
    snapshot_data = []
    for i in range(num_snapshots):
        snapshot = dict(model=dict(idx=i))
        snapshot_path = model_dir / f"snapshot-{i:03}.pt"
        torch.save(snapshot, snapshot_path)
        snapshots.append(Snapshot(best=False, epochs=i, path=snapshot_path))
        snapshot_data.append(snapshot)

    detector_snapshots = []
    detector_data = []
    if task == Task.TOP_DOWN:
        for i in range(num_snapshots):
            snapshot = dict(model=dict(idx=i))
            snapshot_path = model_dir / f"snapshot-detector-{i:03}.pt"
            torch.save(snapshot, snapshot_path)
            detector_data.append(snapshot)
            detector_snapshots.append(
                Snapshot(best=False, epochs=i, path=snapshot_path)
            )

    mock_loader = _make_mock_loader(
        project_path=project_dir,
        project_task="mock",
        project_iteration=0,
        model_folder=model_dir,
        net_type="fake-net",
        pose_task=task,
        default_snapshot_index=-1,
        default_detector_snapshot_index=-1,
    )
    return mock_loader, snapshots, snapshot_data, detector_snapshots, detector_data


@pytest.mark.parametrize(
    "task, num_snapshots, idx, detector_idx",
    [
        (Task.BOTTOM_UP, 10, 0, None),
        (Task.BOTTOM_UP, 10, 5, None),
        (Task.BOTTOM_UP, 10, -1, None),
        (Task.TOP_DOWN, 10, 0, 0),
        (Task.TOP_DOWN, 10, -1, 0),
        (Task.TOP_DOWN, 10, -1, 5),
        (Task.TOP_DOWN, 10, -1, -1),
    ]
)
def test_export_model(
    tmp_path_factory,
    task: Task,
    num_snapshots: int,
    idx: int,
    detector_idx: int | None,
):
    project_dir = tmp_path_factory.mktemp("tmp-project")
    test_data = _get_export_model_data(project_dir, num_snapshots, task)
    mock_loader, snapshots, snapshot_data, detector_snapshots, detector_data = test_data

    def get_mock_loader(*args, **kwargs):
        return mock_loader

    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.export.dlc3_data.DLCLoader",
        get_mock_loader,
    ):
        # export the model
        export.export_model(
            project_dir / "config.yaml",
            snapshotindex=idx,
            detector_snapshot_index=detector_idx,
        )

        # check that the correct snapshot was exported
        snapshot = snapshots[idx]
        detector = None
        if task == Task.TOP_DOWN:
            detector = detector_snapshots[detector_idx]

        dir_name = export.get_export_folder_name(mock_loader)
        filename = export.get_export_filename(mock_loader, snapshot, detector)
        expected_export = project_dir / "exported-models-pytorch" / dir_name / filename
        assert expected_export.exists()

        # check that content of the exports are correct
        exported_data = torch.load(expected_export, weights_only=True)
        assert isinstance(exported_data, dict)
        assert "config" in exported_data
        assert exported_data["config"] == mock_loader.model_cfg

        assert "pose" in exported_data
        assert exported_data["pose"] == snapshot_data[idx]["model"]

        if task == Task.TOP_DOWN:
            assert "detector" in exported_data
            assert exported_data["detector"] == detector_data[detector_idx]["model"]


@patch("deeplabcut.pose_estimation_pytorch.apis.export.wipe_paths_from_model_config")
@pytest.mark.parametrize("task", [Task.BOTTOM_UP, Task.TOP_DOWN])
def test_export_model_clear_paths(mock_wipe: Mock, tmp_path_factory, task: Task):
    project_dir = tmp_path_factory.mktemp("tmp-project")
    test_data = _get_export_model_data(project_dir, 1, task)
    mock_loader, snapshots, snapshot_data, detector_snapshots, detector_data = test_data

    def get_mock_loader(*args, **kwargs):
        return mock_loader

    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.export.dlc3_data.DLCLoader",
        get_mock_loader,
    ):
        export.export_model(project_dir / "config.yaml", wipe_paths=True)

        # check that wipe_paths_from_model_config was called
        assert mock_wipe.call_count == 1
