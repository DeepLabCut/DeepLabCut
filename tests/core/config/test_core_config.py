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
"""Tests for deeplabcut.core.config."""
from pathlib import Path

import pytest

from deeplabcut.core.config import (
    create_config_template,
    create_config_template_3d,
    edit_config,
    pretty_print,
    read_config,
    read_config_as_dict,
    write_config,
    write_config_3d,
    write_config_3d_template,
    write_project_config,
)


# -----------------------------------------------------------------------------
# read_config_as_dict
# -----------------------------------------------------------------------------

def test_read_config_as_dict_loads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 1\nb: [2, 3]\nc:\n  d: 4\n")
    cfg = read_config_as_dict(str(config_path))
    assert cfg == {"a": 1, "b": [2, 3], "c": {"d": 4}}

def test_read_config_as_dict_raises_when_file_missing():
    with pytest.raises(FileNotFoundError):
        read_config_as_dict(Path("/nonexistent/config.yaml"))


def test_read_config_as_dict_raises_on_invalid_yaml(tmp_path):
    """Broken YAML syntax (e.g. unclosed bracket) raises an exception."""
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("key: [unclosed\n")
    with pytest.raises(Exception):
        read_config_as_dict(config_path)


@pytest.mark.skip("This preferred behavior is not yet implemented.")
def test_read_config_as_dict_breaks_for_yaml_tags(tmp_path):
    """read_config breaks for YAML tags like !!python/tuple"""
    config_path = tmp_path / "config.yaml"
    # NOTE @deruyter92 2026-02-03: This test is currently skipped, because 
    # read_config_as_dict currently allows unsafe yaml loading. This should be fixed in the future.
    config_path.write_text(
        "project_path: /old/path\n"
        "engine: pytorch\n"
        "bodyparts: !!python/tuple [a, b, c]\n"
    )
    with pytest.raises(Exception):
        read_config_as_dict(config_path)


def test_read_config_as_dict_accepts_config_with_misnamed_fields(tmp_path):
    """Config with typos/misnamed keys still loads; keys are not validated."""
    # NOTE @deruyter92 2026-02-03: This test captures the current behavior where
    # read_config_as_dict does not validate the keys. This behavior is different 
    # from read_config, which should validate the keys (in the future).
    # The tests can be updated according, depending on the preferred behavior.
    config_path = tmp_path / "typos.yaml"
    config_path.write_text(
        "project_pathh: /wrong\n"  # typo
        "bodypartz: [a, b]\n"      # typo
        "Task: mytask\n"
    )
    cfg = read_config_as_dict(config_path)
    assert cfg["project_pathh"] == "/wrong"
    assert cfg["bodypartz"] == ["a", "b"]
    assert cfg["Task"] == "mytask"
    assert "project_path" not in cfg
    assert "bodyparts" not in cfg
# -----------------------------------------------------------------------------
# write_config
# -----------------------------------------------------------------------------

def test_write_config_creates_file(tmp_path):
    config_path = tmp_path / "out.yaml"
    write_config(config_path, {"a": 1, "b": 2})
    assert config_path.exists()
    cfg = read_config_as_dict(config_path)
    assert cfg["a"] == 1 and cfg["b"] == 2


def test_write_config_overwrites_existing(tmp_path):
    config_path = tmp_path / "out.yaml"
    config_path.write_text("old: true\n")
    write_config(config_path, {"new": True}, overwrite=True)
    cfg = read_config_as_dict(config_path)
    assert "new" in cfg and "old" not in cfg


def test_write_config_raises_when_overwrite_false_and_file_exists(tmp_path):
    config_path = tmp_path / "out.yaml"
    config_path.write_text("x: 1\n")
    with pytest.raises(FileExistsError):
        write_config(config_path, {"x": 2}, overwrite=False)


def test_write_config_allows_overwrite_false_when_file_missing(tmp_path):
    config_path = tmp_path / "new.yaml"
    write_config(config_path, {"x": 1}, overwrite=False)
    assert read_config_as_dict(config_path) == {"x": 1}


# -----------------------------------------------------------------------------
# pretty_print
# -----------------------------------------------------------------------------


def test_pretty_print_flat_config(capsys):
    pretty_print({"a": 1, "b": 2})
    out, _ = capsys.readouterr()
    assert "a: 1" in out and "b: 2" in out


def test_pretty_print_nested_config(capsys):
    pretty_print({"top": {"nested": 42}})
    out, _ = capsys.readouterr()
    assert "top:" in out and "nested: 42" in out


def test_pretty_print_with_indent(capsys):
    pretty_print({"k": "v"}, indent=4)
    out, _ = capsys.readouterr()
    assert out.startswith("    k:")


def test_pretty_print_with_custom_print_fn():
    lines = []

    def capture(s):
        lines.append(s)

    pretty_print({"a": 1}, print_fn=capture)
    assert any("a: 1" in line for line in lines)


# -----------------------------------------------------------------------------
# create_config_template
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("multianimal", [False, True])
def test_create_config_template_returns_tuple(multianimal):
    cfg_file, ruamel_file = create_config_template(multianimal=multianimal)
    assert isinstance(cfg_file, dict)
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
    assert "individuals" not in cfg_file
    assert "uniquebodyparts" not in cfg_file
    assert "multianimalbodyparts" not in cfg_file


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
    assert isinstance(cfg_file_3d, dict)
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
    with pytest.raises(FileNotFoundError, match="Config file at .* not found"):
        read_config(tmp_path / "missing.yaml")


def test_read_config_sets_missing_engine_and_writes_back(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_path: /other/path\n")
    cfg = read_config(config_path)
    assert cfg["engine"] == "tensorflow"
    assert cfg["project_path"] == str(tmp_path)
    # File should have been updated
    cfg_again = read_config_as_dict(config_path)
    assert cfg_again["engine"] == "tensorflow"


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
    assert cfg["project_path"] == str(tmp_path)


def test_read_config_preserves_existing_engine_and_project_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_project_config(
        config_path,
        {"project_path": str(tmp_path), "engine": "pytorch"},
    )
    cfg = read_config(config_path)
    assert cfg["engine"] == "pytorch"
    assert cfg["project_path"] == str(tmp_path)


@pytest.mark.skip("This preferred behavior is not yet implemented.")
def test_read_config_breaks_for_yaml_tags(tmp_path):
    """read_config breaks for YAML tags like !!python/tuple"""
    config_path = tmp_path / "config.yaml"
    # Ruamel fails on !!python/tuple; read_config falls back to PyYAML and repairs the file.
    config_path.write_text(
        "project_path: /old/path\n"
        "engine: pytorch\n"
        "bodyparts: !!python/tuple [a, b, c]\n"
    ) 
    with pytest.raises(Exception):
        read_config(config_path)


@pytest.mark.skip("This preferred behavior is not yet implemented.")
def test_read_config_breaks_for_invalid_fieds(tmp_path):
    # NOTE @deruyter92 2026-02-03: This test is currently skipped, because 
    # read_config does not validate the keys. This should be fixed in the future.
    config_path = tmp_path / "typos.yaml"
    config_path.write_text(
        "project_pathh: /wrong\n"  # typo
        "bodypartz: [a, b]\n"      # typo
        "Task: mytask\n"
    )
    with pytest.raises(Exception):
        read_config(config_path)
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
