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
"""Tests for the configuration versioning and migration system."""
import pytest

from pydantic import ValidationError
from pydantic.dataclasses import dataclass

from deeplabcut.core.config import versioning
from deeplabcut.core.config.versioning import (
    get_config_version,
    migrate_config,
    register_migration,
)
from deeplabcut.core.config.migration_mixin import MigrationMixin
import deeplabcut.core.config.migration_mixin as migration_mixin_module


# -----------------------------------------------------------------------------
# Toy migrations (registered only for tests; v98 <-> v99)
# -----------------------------------------------------------------------------

_TOY_VERSION_OLD = 98
_TOY_VERSION_NEW = 99

_LEGACY_FIELD = "toy_legacy_field"
_NEW_FIELD = "toy_new_field"


@register_migration(_TOY_VERSION_OLD, _TOY_VERSION_NEW)
def _toy_migrate_v98_to_v99(config: dict) -> dict:
    """Test-only: rename toy_legacy_field -> toy_new_field."""
    out = config.copy()
    out["config_version"] = _TOY_VERSION_NEW
    if _LEGACY_FIELD in out:
        out[_NEW_FIELD] = out.pop(_LEGACY_FIELD)
    return out


@register_migration(_TOY_VERSION_NEW, _TOY_VERSION_OLD)
def _toy_migrate_v99_to_v98(config: dict) -> dict:
    """Test-only: rename toy_new_field -> toy_legacy_field."""
    out = config.copy()
    out["config_version"] = _TOY_VERSION_OLD
    if _NEW_FIELD in out:
        out[_LEGACY_FIELD] = out.pop(_NEW_FIELD)
    return out


# -----------------------------------------------------------------------------
# get_config_version
# -----------------------------------------------------------------------------


def test_get_config_version_():
    assert get_config_version({}) == 0
    assert get_config_version({"Task": "mytask"}) == 0
    assert get_config_version({"config_version": 1}) == 1
    assert get_config_version({"config_version": 0}) == 0

# -----------------------------------------------------------------------------
# migrate_config (no-op, errors)
# -----------------------------------------------------------------------------


def test_migrate_config_same_version_returns_unchanged():
    cfg = {"a": 1, "b": 2}
    assert migrate_config(cfg, target_version=0) is cfg

def test_migrate_config_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)
    cfg = {"config_version": _TOY_VERSION_OLD, _LEGACY_FIELD: "x"}
    result = migrate_config(cfg, target_version=_TOY_VERSION_NEW)
    assert cfg[_LEGACY_FIELD] == "x"
    assert result is not cfg


def test_migrate_config_target_exceeds_current_raises(monkeypatch):
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", 99)
    with pytest.raises(ValueError, match="Target version .* exceeds current"):
        migrate_config({"config_version": 0}, target_version=100)


# -----------------------------------------------------------------------------
# Toy migration round-trip (v98 <-> v99)
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _allow_migrate_to_toy_new(monkeypatch):
    """Allow target_version=99 for tests that use the toy v98<->v99 migrations."""
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)


def test_migration_v98_to_v99_renames_toy_field():
    """Upgrade v98 -> v99: toy_legacy_field becomes toy_new_field."""
    v98 = {"config_version": _TOY_VERSION_OLD, _LEGACY_FIELD: "value", "other": 42}
    v99 = migrate_config(v98, target_version=_TOY_VERSION_NEW)
    assert get_config_version(v99) == _TOY_VERSION_NEW
    assert _NEW_FIELD in v99
    assert v99[_NEW_FIELD] == "value"
    assert _LEGACY_FIELD not in v99
    assert v99["other"] == 42


def test_migration_v99_to_v98_renames_toy_field_back():
    """Downgrade v99 -> v98: toy_new_field becomes toy_legacy_field."""
    v99 = {"config_version": _TOY_VERSION_NEW, _NEW_FIELD: "value", "other": 42}
    v98 = migrate_config(v99, target_version=_TOY_VERSION_OLD)
    assert get_config_version(v98) == _TOY_VERSION_OLD
    assert _LEGACY_FIELD in v98
    assert v98[_LEGACY_FIELD] == "value"
    assert _NEW_FIELD not in v98
    assert v98["other"] == 42


def test_roundtrip_v98_to_v99_to_v98_preserves_content():
    """Round-trip v98 -> v99 -> v98: content matches original v98."""
    original = {
        "config_version": _TOY_VERSION_OLD,
        _LEGACY_FIELD: "test_value",
        "Task": "mytask",
        "extra": [1, 2],
    }
    v99 = migrate_config(original, target_version=_TOY_VERSION_NEW)
    back = migrate_config(v99, target_version=_TOY_VERSION_OLD)
    assert get_config_version(back) == _TOY_VERSION_OLD
    assert back[_LEGACY_FIELD] == original[_LEGACY_FIELD]
    assert back["Task"] == original["Task"]
    assert back["extra"] == original["extra"]


def test_roundtrip_v99_to_v98_to_v99_preserves_content():
    """Round-trip v99 -> v98 -> v99: content matches original v99."""
    original = {
        "config_version": _TOY_VERSION_NEW,
        _NEW_FIELD: "test_value",
        "Task": "mytask",
        "extra": [1, 2],
    }
    v98 = migrate_config(original, target_version=_TOY_VERSION_OLD)
    back = migrate_config(v98, target_version=_TOY_VERSION_NEW)
    assert get_config_version(back) == _TOY_VERSION_NEW
    assert back[_NEW_FIELD] == original[_NEW_FIELD]
    assert back["Task"] == original["Task"]
    assert back["extra"] == original["extra"]


def test_roundtrip_v98_without_toy_field_unchanged():
    """Config without toy field is unchanged by v98 -> v99 -> v98."""
    original = {
        "config_version": _TOY_VERSION_OLD,
        "Task": "mytask",
        "bodyparts": ["a", "b"],
    }
    v99 = migrate_config(original, target_version=_TOY_VERSION_NEW)
    back = migrate_config(v99, target_version=_TOY_VERSION_OLD)
    assert get_config_version(back) == _TOY_VERSION_OLD
    assert back["Task"] == original["Task"]
    assert back["bodyparts"] == original["bodyparts"]


# -----------------------------------------------------------------------------
# MigrationMixin: migration prevents validation error for renamed field
# -----------------------------------------------------------------------------


@pytest.fixture
def ToyConfigWithValidField():
    @dataclass(config={'extra': 'forbid'})
    class ToyConfig(MigrationMixin):
        config_version: int = _TOY_VERSION_NEW
        valid_project_config_field: str = ""
    return ToyConfig

def test_config_with_legacy_field_raises_without_migration(monkeypatch, ToyConfigWithValidField):
    """Initializing ProjectConfig with the legacy (wrong) field raises validation error.
    """
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_OLD)
    monkeypatch.setattr(migration_mixin_module, "CURRENT_CONFIG_VERSION", _TOY_VERSION_OLD)
    config_with_legacy_field = {
        "config_version": _TOY_VERSION_OLD,
        "this_fieldname_is_not_in_project_config": "some_value",
    }
    with pytest.raises((ValidationError, TypeError)):
        ToyConfigWithValidField(**config_with_legacy_field)

def test_config_after_migration_accepts_renamed_field(monkeypatch, ToyConfigWithValidField):
    """Registering a migration renames the wrong field so ProjectConfig accepts the config.
    """
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)
    monkeypatch.setattr(migration_mixin_module, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)
    config_with_legacy_field = {
        "config_version": _TOY_VERSION_OLD,
        "this_fieldname_is_not_in_project_config": "some_value",
    }

    @register_migration(_TOY_VERSION_OLD, _TOY_VERSION_NEW)
    def _toy_migrate_legacy_to_valid_field(config: dict) -> dict:
        """Test-only: rename legacy field -> valid_project_config_field."""
        out = config.copy()
        if "this_fieldname_is_not_in_project_config" in out:
            out["valid_project_config_field"] = out.pop("this_fieldname_is_not_in_project_config")
        return out

    cfg = ToyConfigWithValidField(**config_with_legacy_field)
    assert cfg.valid_project_config_field == "some_value"
