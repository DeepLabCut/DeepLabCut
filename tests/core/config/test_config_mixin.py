#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for DLCBaseConfig."""

import pytest
from pydantic import Field, ValidationError

from deeplabcut.core.config import DLCBaseConfig
from deeplabcut.utils.deprecation import DLCDeprecationWarning


class ToyConfig(DLCBaseConfig):
    """Minimal config used to exercise DLCBaseConfig."""

    Task: str = "DefaultTask"
    project_path: str = Field(
        default="DefaultProjectPath",
        json_schema_extra={"aliases": ["projectPath"]},
    )


class NestedInner(DLCBaseConfig):
    lr: float = 0.001
    momentum: float = 0.9


class NestedOuter(DLCBaseConfig):
    name: str = "outer"
    inner: NestedInner | None = Field(default_factory=NestedInner)


# ------------------------------------------------------------------
# Dict-like access protocol
# ------------------------------------------------------------------


class TestGetitem:
    def test_getitem_returns_field_value(self):
        cfg = ToyConfig(Task="t", project_path="/p")
        assert cfg["Task"] == "t"
        assert cfg["project_path"] == "/p"

    def test_getitem_missing_key_raises_key_error(self):
        cfg = ToyConfig()
        with pytest.raises(KeyError):
            cfg["nonexistent"]


class TestSetitem:
    def test_setitem_updates_field(self):
        cfg = ToyConfig()
        cfg["Task"] = "new_task"
        assert cfg.Task == "new_task"
        assert cfg["Task"] == "new_task"

    def test_setitem_unknown_field_raises_key_error(self):
        cfg = ToyConfig()
        with pytest.raises(KeyError, match="no field"):
            cfg["nonexistent"] = 42


class TestContains:
    def test_contains_existing_field(self):
        cfg = ToyConfig()
        assert "Task" in cfg
        assert "project_path" in cfg

    def test_contains_missing_field(self):
        cfg = ToyConfig()
        assert "nonexistent" not in cfg

    def test_contains_non_string_returns_false(self):
        cfg = ToyConfig()
        assert 42 not in cfg


class TestGet:
    def test_get_existing_key(self):
        cfg = ToyConfig(Task="hello")
        assert cfg.get("Task") == "hello"

    def test_get_missing_key_returns_default(self):
        cfg = ToyConfig()
        assert cfg.get("nonexistent") is None
        assert cfg.get("nonexistent", 42) == 42


class TestKeysValuesItems:
    def test_keys(self):
        cfg = ToyConfig(Task="t", project_path="/p")
        assert cfg.keys() == ["Task", "project_path"]

    def test_values(self):
        cfg = ToyConfig(Task="t", project_path="/p")
        assert cfg.values() == ["t", "/p"]

    def test_items(self):
        cfg = ToyConfig(Task="t", project_path="/p")
        assert cfg.items() == [("Task", "t"), ("project_path", "/p")]


class TestIterAndLen:
    def test_iter_yields_keys(self):
        cfg = ToyConfig()
        assert list(cfg) == ["Task", "project_path"]

    def test_len(self):
        cfg = ToyConfig()
        assert len(cfg) == 2

    def test_dict_constructor(self):
        cfg = ToyConfig(Task="t", project_path="/p")
        d = dict(cfg.items())
        assert d == {"Task": "t", "project_path": "/p"}


# ------------------------------------------------------------------
# select() utility
# ------------------------------------------------------------------


class TestSelect:
    def test_select_single_level(self):
        cfg = ToyConfig(Task="t")
        assert cfg.select("Task") == "t"

    def test_select_nested_config(self):
        cfg = NestedOuter(name="outer", inner=NestedInner(lr=0.01))
        assert cfg.select("inner.lr") == 0.01
        assert cfg.select("inner.momentum") == 0.9

    def test_select_missing_returns_default(self):
        cfg = ToyConfig()
        assert cfg.select("nonexistent") is None
        assert cfg.select("nonexistent", "fallback") == "fallback"

    def test_select_deep_missing_returns_default(self):
        cfg = NestedOuter()
        assert cfg.select("inner.nonexistent") is None
        assert cfg.select("nope.deep.path", 0) == 0

    def test_select_none_intermediate(self):
        cfg = NestedOuter(inner=None)
        assert cfg.select("inner.lr") is None


# ------------------------------------------------------------------
# model_validate / extra=forbid
# ------------------------------------------------------------------


class TestModelValidate:
    def test_model_validate_returns_instance(self):
        cfg = ToyConfig.model_validate({"Task": "task", "project_path": ""})
        assert isinstance(cfg, ToyConfig)
        assert cfg.Task == "task"

    def test_model_validate_rejects_extra_keys(self):
        with pytest.raises(ValidationError):
            ToyConfig.model_validate({"Task": "ok", "project_path": "", "extra": 1})


# ------------------------------------------------------------------
# aliases
# ------------------------------------------------------------------


class TestAliases:
    def test_model_validate_resolves_alias_without_from_dict(self):
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            cfg = ToyConfig.model_validate({"Task": "t", "projectPath": "/alias"})
        assert cfg.project_path == "/alias"

    def test_model_validate_rejects_alias_and_canonical_together(self):
        with pytest.raises(TypeError, match="both 'projectPath' and 'project_path'"):
            ToyConfig.model_validate({"Task": "t", "projectPath": "/alias", "project_path": "/canonical"})

    def test_getitem_alias_warns_and_reads_canonical(self):
        cfg = ToyConfig(project_path="/p")
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            assert cfg["projectPath"] == "/p"

    def test_setitem_alias_warns_once_and_writes_canonical(self):
        cfg = ToyConfig()
        with pytest.warns(DLCDeprecationWarning, match="projectPath") as record:
            cfg["projectPath"] = "/new"
        assert len(record) == 1
        assert cfg.project_path == "/new"

    def test_setattr_alias_warns_and_validates(self):
        cfg = ToyConfig()
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            cfg.projectPath = "/attr"
        assert cfg.project_path == "/attr"

    def test_getattr_alias_warns(self):
        cfg = ToyConfig(project_path="/p")
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            assert cfg.projectPath == "/p"

    def test_contains_accepts_alias(self):
        cfg = ToyConfig()
        assert "projectPath" in cfg


# ------------------------------------------------------------------
# from_dict / from_any / from_yaml / to_dict / to_yaml
# ------------------------------------------------------------------


def test_from_dict_returns_instance():
    cfg = ToyConfig.from_dict({"Task": "t", "project_path": "/p"})
    assert isinstance(cfg, ToyConfig)
    assert cfg.Task == "t"
    assert cfg.project_path == "/p"


def test_from_dict_incomplete_uses_defaults():
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
