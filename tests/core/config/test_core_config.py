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
"""Tests for the config helpers DeepLabCut keeps; the rest are tested in dlccore."""

import logging
from collections.abc import Mapping

import pytest
from pydantic import ValidationError
from ruamel.yaml.error import YAMLError

from deeplabcut.core.config import (
    create_config_template,
    create_config_template_3d,
    edit_config,
    read_config,
    read_config_as_dict,
    write_config_3d,
    write_config_3d_template,
    write_project_config,
)
from deeplabcut.core.deprecation import DLCDeprecationWarning

# -----------------------------------------------------------------------------
# create_config_template
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("multianimal", [False, True])
def test_create_config_template_returns_tuple(multianimal):
    cfg_file, ruamel_file = create_config_template(multianimal=multianimal)
    assert isinstance(cfg_file, Mapping)
    assert ruamel_file is not None


def test_create_config_template_single_animal_has_expected_keys():
    cfg_file, _ = create_config_template(multianimal=False)
    assert "Task" in cfg_file
    assert "project_path" in cfg_file
    assert "bodyparts" in cfg_file
    assert "engine" in cfg_file
    assert "video_sets" in cfg_file
    assert "multianimalproject" in cfg_file
    assert "detector_batch_size" in cfg_file


def test_create_config_template_multianimal_has_extra_keys():
    cfg_file, _ = create_config_template(multianimal=True)
    assert "individuals" in cfg_file
    assert "uniquebodyparts" in cfg_file
    assert "multianimalbodyparts" in cfg_file
    assert "bodyparts" in cfg_file


# -----------------------------------------------------------------------------
# create_config_template_3d
# -----------------------------------------------------------------------------


def test_create_config_template_3d_returns_tuple():
    cfg_file_3d, ruamel_file_3d = create_config_template_3d()
    assert isinstance(cfg_file_3d, Mapping)
    assert ruamel_file_3d is not None


def test_create_config_template_3d_has_expected_keys():
    cfg_file, _ = create_config_template_3d()
    assert "Task" in cfg_file
    assert "project_path" in cfg_file
    assert "skeleton" in cfg_file
    assert "num_cameras" in cfg_file
    assert "camera_names" in cfg_file
    assert "scorername_3d" in cfg_file


# -----------------------------------------------------------------------------
# read_config
# -----------------------------------------------------------------------------


def test_read_config_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_config(tmp_path / "missing.yaml")


def test_read_config_sets_missing_engine_and_writes_back(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_path: /other/path\n")
    cfg = read_config(config_path)
    assert cfg["engine"] == "pytorch"
    assert cfg["project_path"] == tmp_path
    # File should have been updated
    cfg_again = read_config_as_dict(config_path)
    assert cfg_again["engine"] == "pytorch"


def test_read_config_sets_detector_snapshotindex_when_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(config_path, {"project_path": str(tmp_path), "engine": "pytorch"})
    cfg = read_config(config_path)
    assert cfg["detector_snapshotindex"] == -1


def test_read_config_sets_detector_batch_size_when_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(config_path, {"project_path": str(tmp_path), "engine": "pytorch"})
    cfg = read_config(config_path)
    assert cfg["detector_batch_size"] == 1


def test_read_config_updates_project_path_when_different(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": "/old/path", "engine": "pytorch"},
    )
    cfg = read_config(config_path)
    assert cfg["project_path"] == tmp_path


@pytest.mark.parametrize("engine", ["pytorch", "tensorflow"])
def test_read_config_preserves_existing_engine_and_project_path(tmp_path, engine):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": str(tmp_path), "engine": engine},
    )
    cfg = read_config(config_path)
    assert cfg["engine"] == engine
    assert cfg["project_path"] == tmp_path


def test_read_config_breaks_for_yaml_tags(tmp_path):
    """read_config raises for YAML files containing unsafe tags like !!python/tuple."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_path: /old/path\nengine: pytorch\nbodyparts: !!python/tuple [a, b, c]\n")
    with pytest.raises(YAMLError):
        read_config(config_path)


def test_read_config_breaks_for_invalid_fields(tmp_path):
    """read_config raises when the YAML contains field names not declared by ProjectConfig."""
    config_path = tmp_path / "typos.yaml"
    config_path.write_text(
        "project_pathh: /wrong\n"  # typo
        "bodypartz: [a, b]\n"  # typo
        "Task: mytask\n"
    )
    with pytest.raises(ValidationError):
        read_config(config_path)


def test_read_config_loads_null_non_optional_fields(tmp_path):
    """Bare/null keys in config.yaml are treated as unset (legacy template support)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"project_path: {tmp_path}\nengine: pytorch\nmultianimalproject:\nidentity:\n")
    cfg = read_config(config_path)
    assert cfg["multianimalproject"] is False
    assert cfg["identity"] is None


@pytest.mark.parametrize("ignore_empty", [True, False])
def test_read_config_ignore_empty_is_deprecated_noop(tmp_path, ignore_empty):
    """Any explicit ignore_empty=... warns and has no effect; null keys use defaults."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"project_path: {tmp_path}\nengine: pytorch\nmultianimalproject:\n")
    with pytest.warns(DLCDeprecationWarning, match="ignore_empty"):
        cfg = read_config(config_path, ignore_empty=ignore_empty)
    assert cfg["multianimalproject"] is False


# -----------------------------------------------------------------------------
# write_project_config
# -----------------------------------------------------------------------------


def test_write_project_config_writes_valid_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": str(tmp_path), "Task": "mytask", "bodyparts": ["a", "b"]},
    )
    cfg = read_config_as_dict(config_path)
    assert cfg["project_path"] == str(tmp_path)
    assert cfg["Task"] == "mytask"
    assert cfg["bodyparts"] == ["a", "b"]


def test_write_project_config_adds_skeleton_defaults_when_missing(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": str(tmp_path), "multianimalproject": False},
    )
    cfg = read_config_as_dict(config_path)
    assert cfg["skeleton"] == []
    assert cfg["skeleton_color"] == "black"


def test_write_project_config_uses_multianimal_template_when_flag_true(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": str(tmp_path), "multianimalproject": True},
    )
    cfg = read_config_as_dict(config_path)
    assert "individuals" in cfg
    assert "uniquebodyparts" in cfg


def test_write_project_config_falls_back_and_preserves_unknown_keys(tmp_path, caplog):
    """Unvalidated legacy write succeeds via fallback with logging, but may not round-trip via read_config."""
    config_path = tmp_path / "config.yaml"
    payload = {
        "project_path": str(tmp_path),
        "Task": "mytask",
        "bodyparts": ["a", "b"],
        "legacy_custom_field": "keep-me",  # extra="forbid" on ProjectConfig, will be preserved
    }

    with caplog.at_level(logging.ERROR, logger="deeplabcut.core.config.utils"):
        with pytest.warns(UserWarning, match="legacy config file writing"):
            write_project_config(config_path, payload)

    assert "Invalid configuration" in caplog.text
    assert config_path.is_file()
    written = read_config_as_dict(config_path)
    assert written["legacy_custom_field"] == "keep-me"
    assert written["Task"] == "mytask"

    with pytest.raises(ValidationError):
        read_config(config_path)


# -----------------------------------------------------------------------------
# edit_config
# -----------------------------------------------------------------------------


def test_edit_config_applies_edits_and_overwrites_original(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\nb: 2\n")
    cfg = edit_config(config_path, {"b": 99, "c": 3})
    assert cfg["a"] == 1 and cfg["b"] == 99 and cfg["c"] == 3
    loaded = read_config_as_dict(config_path)
    assert loaded["b"] == 99 and loaded["c"] == 3


def test_edit_config_writes_to_output_name_when_given(tmp_path):
    src = tmp_path / "src.yaml"
    src.write_text("x: 1\n")
    out = tmp_path / "out.yaml"
    cfg = edit_config(src, {"x": 2}, output_name=out)
    assert cfg["x"] == 2
    assert read_config_as_dict(out)["x"] == 2
    assert read_config_as_dict(src)["x"] == 1


# -----------------------------------------------------------------------------
# write_config_3d
# -----------------------------------------------------------------------------


def test_write_config_3d_writes_valid_yaml(tmp_path):
    config_path = tmp_path / "config_3d.yaml"
    write_config_3d(
        config_path,
        {"project_path": str(tmp_path), "Task": "3dtask", "num_cameras": 2},
    )
    cfg = read_config_as_dict(config_path)
    assert cfg["project_path"] == str(tmp_path)
    assert cfg["Task"] == "3dtask"
    assert cfg["num_cameras"] == 2


# -----------------------------------------------------------------------------
# write_config_3d_template
# -----------------------------------------------------------------------------


def test_write_config_3d_template_writes_given_template(tmp_path):
    config_path = tmp_path / "config_3d.yaml"
    cfg_file, ruamel_file = create_config_template_3d()
    cfg_file["Task"] = "custom_3d"
    cfg_file["num_cameras"] = 3
    write_config_3d_template(config_path, cfg_file, ruamel_file)
    cfg = read_config_as_dict(config_path)
    assert cfg["Task"] == "custom_3d"
    assert cfg["num_cameras"] == 3
