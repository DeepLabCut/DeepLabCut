#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for ConfigMixin."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic.dataclasses import dataclass

from deeplabcut.core.config import ConfigMixin


@dataclass
class ToyConfig(ConfigMixin):
    """Minimal config used to exercise ConfigMixin."""
    Task: str = "DefaultTask"
    project_path: str = "DefaultProjectPath"


def test_validate_dict_returns_dictconfig():
    cfg = ToyConfig.validate_dict({"Task": "task", "project_path": ""})
    assert OmegaConf.is_config(cfg)
    assert cfg.Task == "task"


def test_from_dict_returns_instance():
    cfg = ToyConfig.from_dict({"Task": "t", "project_path": "/p"})
    assert isinstance(cfg, ToyConfig)
    assert cfg.Task == "t"
    assert cfg.project_path == "/p"


def test_from_dict_incomplete_uses_defaults():
    """Missing keys in dict input are filled from ToyConfig defaults."""
    cfg = ToyConfig.from_dict({"Task": "custom_task"})
    assert cfg.Task == "custom_task"
    assert cfg.project_path == "DefaultProjectPath"

    cfg_empty = ToyConfig.from_dict({})
    assert cfg_empty.Task == "DefaultTask"
    assert cfg_empty.project_path == "DefaultProjectPath"


def test_from_any_with_instance_returns_same():
    cfg = ToyConfig.from_dict({"Task": "x", "project_path": ""})
    out = ToyConfig.from_any(cfg)
    assert out is cfg


def test_from_any_with_dict_returns_instance():
    cfg = ToyConfig.from_any({"Task": "y", "project_path": ""})
    assert isinstance(cfg, ToyConfig)
    assert cfg.Task == "y"


def test_from_any_with_invalid_type_raises():
    with pytest.raises(TypeError, match="Expected.*Got <class 'int'>"):
        ToyConfig.from_any(42)


def test_from_yaml_loads_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("Task: mytask\nproject_path: /foo\n")
    cfg = ToyConfig.from_yaml(path)
    assert cfg.Task == "mytask"
    assert cfg.project_path == "/foo"


def test_to_dict_returns_dict():
    cfg = ToyConfig.from_dict({"Task": "t", "project_path": ""})
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["Task"] == "t"


def test_to_dictconfig_returns_dictconfig():
    cfg = ToyConfig.from_dict({"Task": "t", "project_path": ""})
    dc = cfg.to_dictconfig()
    assert OmegaConf.is_config(dc)
    assert dc.Task == "t"


def test_from_dict_to_dict_roundtrip():
    d = {"Task": "rt", "project_path": "/x"}
    cfg = ToyConfig.from_dict(d)
    out = cfg.to_dict()
    restored = ToyConfig.from_dict(out)
    assert restored.Task == cfg.Task
    assert restored.project_path == cfg.project_path


def test_to_yaml_roundtrip(tmp_path):
    path = tmp_path / "out.yaml"
    orig = ToyConfig.from_dict({"Task": "rt", "project_path": "/x"})
    orig.to_yaml(path)
    loaded = ToyConfig.from_yaml(path)
    assert loaded.Task == orig.Task
    assert loaded.project_path == orig.project_path


def test_print_no_error(capsys):
    cfg = ToyConfig.from_dict({"Task": "p", "project_path": ""})
    cfg.print()
    out, _ = capsys.readouterr()
    assert "Task" in out
