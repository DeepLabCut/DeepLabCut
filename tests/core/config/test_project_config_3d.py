#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for ProjectConfig3D, the config of a 3D project."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from deeplabcut.core.config import CameraConfig3D, ProjectConfig, ProjectConfig3D, read_config_as_dict, write_config


@pytest.fixture
def flat_config():
    """A minimal 3D config.yaml, with the per-camera settings as flat keys."""
    return {
        "Task": "trial",
        "scorer": "tester",
        "date": "Nov13",
        "project_path": "/old/location/trial-tester-3d",
        "num_cameras": 2,
        "camera_names": ["camera-1", "camera-2"],
        "scorername_3d": "DLC_3D",
        "config_file_camera-1": "/2d/cam1/config.yaml",
        "config_file_camera-2": "/2d/cam2/config.yaml",
        "shuffle_camera-1": 1,
        "trainingsetindex_camera-1": 0,
        "shuffle_camera-2": 2,
        "trainingsetindex_camera-2": 1,
    }


def test_groups_flat_camera_keys(flat_config):
    cfg = ProjectConfig3D.from_dict(flat_config)

    assert cfg.camera_names == ["camera-1", "camera-2"]
    assert cfg.cameras["camera-2"].shuffle == 2
    assert cfg.cameras["camera-2"].trainingsetindex == 1


def test_yaml_round_trip_keeps_flat_layout(flat_config, tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path, flat_config)

    cfg = ProjectConfig3D.from_yaml(path)
    cfg.to_yaml(path)

    on_disk = read_config_as_dict(path)
    assert set(flat_config) <= set(on_disk)
    assert "cameras" not in on_disk
    assert ProjectConfig3D.from_yaml(path) == cfg


def test_from_yaml_repairs_project_path(flat_config, tmp_path):
    path = tmp_path / "config.yaml"
    write_config(path, flat_config)

    cfg = ProjectConfig3D.from_yaml(path)

    assert cfg.project_path == tmp_path.absolute()
    assert "project_path" in cfg.dirty_fields


def test_rename_and_add_cameras(flat_config):
    cfg = ProjectConfig3D.from_dict(flat_config)

    cfg.cameras = {"left": cfg.cameras["camera-1"], "right": cfg.cameras["camera-2"]}
    cfg.cameras["top"] = CameraConfig3D(shuffle=3)

    dumped = cfg.to_dict()
    assert dumped["camera_names"] == ["left", "right", "top"]
    assert (dumped["shuffle_right"], dumped["shuffle_top"]) == (2, 3)


def test_renamed_keys_take_precedence_over_template_keys(flat_config):
    """calibrate_cameras used to rename config_file_/shuffle_ keys to the camera names, but not
    trainingsetindex_. Renamed keys win; the template's camera-<i> keys fill in the rest."""
    flat_config["camera_names"] = ["left", "right"]
    flat_config["config_file_left"] = "/2d/left/config.yaml"  # renamed, template key still present
    flat_config["shuffle_left"] = flat_config.pop("shuffle_camera-1")

    cfg = ProjectConfig3D.from_dict(flat_config)

    assert cfg.camera_names == ["left", "right"]
    assert cfg.cameras["left"].config_file == Path("/2d/left/config.yaml")
    assert cfg.cameras["left"].shuffle == 1
    assert cfg.cameras["right"].trainingsetindex == 1


def test_ignores_keys_of_the_2d_project_template(flat_config):
    """DeepLabCut <= 3.0.0 could write 3D configs back with the 2D project template keys."""
    flat_config.update(engine="pytorch", bodyparts=["a", "b"], TrainingFraction=[0.95], alphavalue=0.7)

    dumped = ProjectConfig3D.from_dict(flat_config).to_dict()

    assert not {"engine", "bodyparts", "TrainingFraction", "alphavalue"} & dumped.keys()


def test_explains_2d_project_config():
    """A 2D config passed to a 3D function must say so, not list every 2D key as an extra input."""
    with pytest.raises(ValidationError, match="config.yaml of a 2D project where .* 3D project is expected"):
        ProjectConfig3D.from_dict(ProjectConfig().to_dict())


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"unknown_key": 1}, "unknown_key.*Extra inputs are not permitted"),
        ({"shuffle_camera-3": 1}, "shuffle_camera-3.*Extra inputs are not permitted"),
        ({"shuffle_camera-1": -1}, "greater than or equal to 0"),
    ],
    ids=["unknown_key", "unknown_camera", "invalid_shuffle"],
)
def test_rejects_invalid_configs(flat_config, extra, match):
    with pytest.raises(ValidationError, match=match):
        ProjectConfig3D.from_dict({**flat_config, **extra})
