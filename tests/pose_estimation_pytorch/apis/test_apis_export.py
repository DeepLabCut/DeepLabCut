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
import copy
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.apis.export as export
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.pose_estimation_pytorch import Task
from deeplabcut.pose_estimation_pytorch.runners.snapshots import Snapshot


@pytest.fixture()
def project_dir(tmp_path_factory) -> Path:
    project_dir = tmp_path_factory.mktemp("tmp-project")
    print(f"\nTemporary project directory:")
    print(str(project_dir))
    print("---")
    yield project_dir
    shutil.rmtree(str(project_dir))


def _mock_multianimal_project(project_dir: Path):
    video_dir = project_dir / "videos"
    video_dir.mkdir(exist_ok=True)

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
        project_path=str(project_path),
        Task=project_task,
        date="Jan12",
        TrainingFraction=[0.95],
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


def _get_export_model_data(
    project_dir: Path,
    num_snapshots: int,
    task: Task,
    project_iteration: int = 0,
):
    _mock_multianimal_project(project_dir)

    model_dir = Path(project_dir) / f"iteration-{project_iteration}" / "fake-shuffle-0"
    model_dir.mkdir(exist_ok=True, parents=True)
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
        project_iteration=project_iteration,
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
    ],
)
def test_export_model(
    project_dir,
    task: Task,
    num_snapshots: int,
    idx: int,
    detector_idx: int | None,
):
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
def test_export_model_clear_paths(mock_wipe: Mock, project_dir, task: Task):
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


@pytest.mark.parametrize("task", [Task.BOTTOM_UP, Task.TOP_DOWN])
@pytest.mark.parametrize("overwrite", [True, False])
def test_export_overwrite(project_dir, task: Task, overwrite: bool):
    test_data = _get_export_model_data(project_dir, 1, task)
    mock_loader, snapshots, snapshot_data, detector_snapshots, detector_data = test_data
    snapshot = snapshots[0]
    detector = None if task == Task.BOTTOM_UP else detector_snapshots[0]

    def get_mock_loader(*args, **kwargs):
        return mock_loader

    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.export.dlc3_data.DLCLoader",
        get_mock_loader,
    ):
        dir_name = export.get_export_folder_name(mock_loader)
        filename = export.get_export_filename(mock_loader, snapshot, detector)
        expected_export = project_dir / "exported-models-pytorch" / dir_name / filename
        expected_export.parent.mkdir(exist_ok=False, parents=True)

        # add existing data
        assert not expected_export.exists()
        existing_data = dict()
        torch.save(existing_data, expected_export)

        # export data
        export.export_model(project_dir / "config.yaml", overwrite=overwrite)

        exported_data = torch.load(expected_export, weights_only=True)

        if overwrite:
            assert existing_data != exported_data
        else:
            assert existing_data == exported_data


@pytest.mark.parametrize("task", [Task.BOTTOM_UP, Task.TOP_DOWN])
@pytest.mark.parametrize("iteration", [5, 12])
def test_export_change_iteration(project_dir, task: Task, iteration: int):
    test_data = _get_export_model_data(
        project_dir,
        1,
        task,
        project_iteration=0,
    )
    mock_loader, snapshots, snapshot_data, detector_snapshots, detector_data = test_data
    snapshot = snapshots[0]
    detector = None if task == Task.BOTTOM_UP else detector_snapshots[0]

    loader_diff_iter = _get_export_model_data(
        project_dir, 1, task, project_iteration=iteration
    )[0]

    def get_mock_loader(config, *args, **kwargs):
        _loader = copy.deepcopy(mock_loader)
        if isinstance(config, dict):
            _loader = copy.deepcopy(mock_loader)
            _loader.project_cfg = config
        return _loader

    def read_mock_config(*args, **kwargs):
        return copy.deepcopy(mock_loader.project_cfg)

    # patch the DLCLoader but also read_config
    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.export.dlc3_data.DLCLoader",
        get_mock_loader,
    ):
        with patch(
            "deeplabcut.pose_estimation_pytorch.apis.export.af.read_config",
            read_mock_config,
        ):
            # check no exports exist yet
            for loader in [mock_loader, loader_diff_iter]:
                dir_name = export.get_export_folder_name(loader)
                filename = export.get_export_filename(loader, snapshot, detector)
                assert not (
                    project_dir / "exported-models-pytorch" / dir_name / filename
                ).exists()

            # export data
            export.export_model(project_dir / "config.yaml", iteration=iteration)

            # check the export exists for the correct iteration
            for loader, file_should_exist in [
                (mock_loader, False),
                (loader_diff_iter, True),
            ]:
                dir_name = export.get_export_folder_name(loader)
                filename = export.get_export_filename(loader, snapshot, detector)
                expected = project_dir / "exported-models-pytorch" / dir_name / filename
                expected_exists = expected.exists()
                assert expected_exists == file_should_exist
