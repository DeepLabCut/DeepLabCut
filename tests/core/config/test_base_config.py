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

from deeplabcut.core.config import DLCBaseConfig, DLCVersionedConfig, versioning
from deeplabcut.core.config.versioning import register_migration
from deeplabcut.utils.deprecation import DLCDeprecationWarning

_TOY_VERSION_OLD = 98
_TOY_VERSION_NEW = 99
_LEGACY_FIELD = "toy_legacy_field"


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

    def test_keys_does_not_include_alias(self):
        """keys() / iter() only yield canonical field names, never alias names."""
        cfg = ToyConfig()
        assert "projectPath" not in cfg.keys()
        assert "projectPath" not in list(cfg)

    def test_to_dict_uses_canonical_names_only(self):
        """to_dict() always emits canonical field names, even when loaded via alias."""
        with pytest.warns(DLCDeprecationWarning):
            cfg = ToyConfig.model_validate({"Task": "t", "projectPath": "/alias"})
        d = cfg.to_dict()
        assert "project_path" in d
        assert "projectPath" not in d

    def test_from_yaml_with_alias_key_warns_and_loads(self, tmp_path):
        """A YAML file that contains an alias key emits a warning on load."""
        path = tmp_path / "config.yaml"
        path.write_text("Task: t\nprojectPath: /from_yaml\n")
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            cfg = ToyConfig.from_yaml(path)
        assert cfg.project_path == "/from_yaml"

    def test_to_yaml_then_from_yaml_no_alias_warning(self, tmp_path):
        """After saving via to_yaml, reloading must not emit any alias warning
        (the YAML file should contain canonical field names)."""
        import warnings

        path = tmp_path / "config.yaml"
        cfg = ToyConfig(Task="t", project_path="/p")
        cfg.to_yaml(path)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DLCDeprecationWarning)
            loaded = ToyConfig.from_yaml(path)
        assert loaded.project_path == "/p"


# ------------------------------------------------------------------
# validate_assignment on DLCBaseConfig (not just versioned)
# ------------------------------------------------------------------


class TypedBase(DLCBaseConfig):
    count: int = 0


class TestBaseConfigValidateAssignment:
    def test_invalid_assignment_raises(self):
        cfg = TypedBase()
        with pytest.raises(ValidationError):
            cfg.count = "not-a-number"
        assert cfg.count == 0

    def test_valid_assignment_takes_effect(self):
        cfg = TypedBase()
        cfg.count = 7
        assert cfg.count == 7

    def test_coercion_applied(self):
        cfg = TypedBase()
        cfg.count = True  # coerced to 1
        assert cfg.count == 1


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


# ------------------------------------------------------------------
# No schema migration on DLCBaseConfig
# ------------------------------------------------------------------


class _ToyBaseOnly(DLCBaseConfig):
    toy_new_field: str = ""


class _ToyVersioned(DLCVersionedConfig):
    config_version: int = _TOY_VERSION_NEW
    toy_new_field: str = ""


@register_migration(_TOY_VERSION_OLD, _TOY_VERSION_NEW, config_type="_ToyVersioned")
def _toy_versioned_migrate_v98_to_v99(config: dict) -> dict:
    if _LEGACY_FIELD in config:
        config["toy_new_field"] = config.pop(_LEGACY_FIELD)
    return config


class TestNoMigrationOnBaseConfig:
    """DLCBaseConfig resolves aliases but does not run version migrations."""

    @pytest.fixture(autouse=True)
    def _toy_current_version(self, monkeypatch):
        monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)

    def test_base_config_rejects_legacy_key_without_migration(self):
        legacy_cfg = {
            "config_version": _TOY_VERSION_OLD,
            _LEGACY_FIELD: "value",
        }
        with pytest.raises(ValidationError):
            _ToyBaseOnly.model_validate(legacy_cfg)

    def test_versioned_config_applies_migration_for_same_input(self):
        legacy_cfg = {
            "config_version": _TOY_VERSION_OLD,
            _LEGACY_FIELD: "value",
        }
        cfg = _ToyVersioned.model_validate(legacy_cfg)
        assert cfg.toy_new_field == "value"
