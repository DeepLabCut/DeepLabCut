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
import logging

import pytest

from pydantic import ConfigDict, ValidationError
from pydantic.dataclasses import dataclass

from deeplabcut.core.config import versioning
from deeplabcut.core.config.versioning import (
    get_config_version,
    migrate_config,
    register_migration,
)
from deeplabcut.core.config.versioning import MigrationMixin

_LOGGER_NAME = "deeplabcut.core.config.versioning"


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
def _isolated_toy_migration(monkeypatch):
    """Isolate each test from global migration registry changes."""
    monkeypatch.setattr(versioning, "_MIGRATIONS", versioning._MIGRATIONS.copy())
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
    config_with_legacy_field = {
        "config_version": _TOY_VERSION_OLD,
        "this_fieldname_is_not_in_project_config": "some_value",
    }

    # Replace the existing toy (98→99) migration with one that renames
    # the unknown field to a field the dataclass actually declares.
    del versioning._MIGRATIONS[(_TOY_VERSION_OLD, _TOY_VERSION_NEW)]

    @register_migration(_TOY_VERSION_OLD, _TOY_VERSION_NEW)
    def _toy_migrate_legacy_to_valid_field(config: dict) -> dict:
        """Test-only: rename legacy field -> valid_project_config_field."""
        if "this_fieldname_is_not_in_project_config" in config:
            config["valid_project_config_field"] = config.pop("this_fieldname_is_not_in_project_config")
        return config

    cfg = ToyConfigWithValidField(**config_with_legacy_field)
    assert cfg.valid_project_config_field == "some_value"


# =============================================================================
# Multi-step migration chain (v50 → v51 → v52 → v53)
#
# Each step renames one field, so we can verify individual steps, full chains,
# downgrades, missing intermediates, missing fields, and mid-chain errors.
# =============================================================================

_V50, _V51, _V52, _V53 = 50, 51, 52, 53


def _register_chain_migrations():
    """Register a v50↔v53 migration chain (upgrades and downgrades)."""
    @register_migration(_V50, _V51)
    def _chain_v50_to_v51(config: dict) -> dict:
        if "field_a_old" in config:
            config["field_a_new"] = config.pop("field_a_old")
        return config

    @register_migration(_V51, _V52)
    def _chain_v51_to_v52(config: dict) -> dict:
        if "field_b_old" in config:
            config["field_b_new"] = config.pop("field_b_old")
        return config

    @register_migration(_V52, _V53)
    def _chain_v52_to_v53(config: dict) -> dict:
        if "field_c_old" in config:
            config["field_c_new"] = config.pop("field_c_old")
        return config

    @register_migration(_V51, _V50)
    def _chain_v51_to_v50(config: dict) -> dict:
        if "field_a_new" in config:
            config["field_a_old"] = config.pop("field_a_new")
        return config

    @register_migration(_V52, _V51)
    def _chain_v52_to_v51(config: dict) -> dict:
        if "field_b_new" in config:
            config["field_b_old"] = config.pop("field_b_new")
        return config

    @register_migration(_V53, _V52)
    def _chain_v53_to_v52(config: dict) -> dict:
        if "field_c_new" in config:
            config["field_c_old"] = config.pop("field_c_new")
        return config


@pytest.fixture
def chain_migrations(monkeypatch):
    """Register v50↔v53 chain and set CURRENT_CONFIG_VERSION=53."""
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _V53)
    _register_chain_migrations()


# -----------------------------------------------------------------------------
# Multi-step upgrade chains
# -----------------------------------------------------------------------------


class TestMultiStepUpgrade:
    """Tests for chained upgrade migrations across multiple versions."""

    def test_upgrade_v50_to_v53_applies_all_steps(self, chain_migrations):
        """Full chain v50→v53 renames all three fields."""
        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
            "untouched": 42,
        }
        result = migrate_config(cfg, target_version=_V53)
        assert result["config_version"] == _V53
        assert result["field_a_new"] == "a"
        assert result["field_b_new"] == "b"
        assert result["field_c_new"] == "c"
        assert result["untouched"] == 42
        assert "field_a_old" not in result
        assert "field_b_old" not in result
        assert "field_c_old" not in result

    def test_upgrade_v50_to_v52_partial_chain(self, chain_migrations):
        """Partial chain v50→v52 renames only fields a and b."""
        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
        }
        result = migrate_config(cfg, target_version=_V52)
        assert result["config_version"] == _V52
        assert result["field_a_new"] == "a"
        assert result["field_b_new"] == "b"
        assert result["field_c_old"] == "c"  # untouched — v52→v53 not applied

    def test_upgrade_single_step_v51_to_v52(self, chain_migrations):
        """Single step in the middle of the chain works."""
        cfg = {"config_version": _V51, "field_b_old": "b"}
        result = migrate_config(cfg, target_version=_V52)
        assert result["config_version"] == _V52
        assert result["field_b_new"] == "b"

    def test_upgrade_does_not_mutate_original(self, chain_migrations):
        """Original config dict is untouched after multi-step upgrade."""
        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
        }
        original_copy = cfg.copy()
        migrate_config(cfg, target_version=_V53)
        assert cfg == original_copy


# -----------------------------------------------------------------------------
# Multi-step downgrade chains
# -----------------------------------------------------------------------------


class TestMultiStepDowngrade:
    """Tests for chained downgrade migrations across multiple versions."""

    def test_downgrade_v53_to_v50_applies_all_steps(self, chain_migrations):
        """Full downgrade v53→v50 reverses all three field renames."""
        cfg = {
            "config_version": _V53,
            "field_a_new": "a",
            "field_b_new": "b",
            "field_c_new": "c",
            "untouched": 42,
        }
        result = migrate_config(cfg, target_version=_V50)
        assert result["config_version"] == _V50
        assert result["field_a_old"] == "a"
        assert result["field_b_old"] == "b"
        assert result["field_c_old"] == "c"
        assert result["untouched"] == 42

    def test_downgrade_v53_to_v51_partial(self, chain_migrations):
        """Partial downgrade v53→v51 reverses only fields c and b."""
        cfg = {
            "config_version": _V53,
            "field_a_new": "a",
            "field_b_new": "b",
            "field_c_new": "c",
        }
        result = migrate_config(cfg, target_version=_V51)
        assert result["config_version"] == _V51
        assert result["field_a_new"] == "a"  # untouched — v51→v50 not applied
        assert result["field_b_old"] == "b"
        assert result["field_c_old"] == "c"

    def test_downgrade_does_not_mutate_original(self, chain_migrations):
        """Original config dict is untouched after multi-step downgrade."""
        cfg = {
            "config_version": _V53,
            "field_a_new": "a",
            "field_b_new": "b",
            "field_c_new": "c",
        }
        original_copy = cfg.copy()
        migrate_config(cfg, target_version=_V50)
        assert cfg == original_copy


# -----------------------------------------------------------------------------
# Multi-step round-trips
# -----------------------------------------------------------------------------


class TestMultiStepRoundTrip:
    """Verify that upgrade→downgrade and downgrade→upgrade round-trips preserve data."""

    def test_roundtrip_v50_to_v53_and_back(self, chain_migrations):
        original = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
            "extra": [1, 2, 3],
        }
        upgraded = migrate_config(original, target_version=_V53)
        back = migrate_config(upgraded, target_version=_V50)
        assert back["config_version"] == _V50
        assert back["field_a_old"] == "a"
        assert back["field_b_old"] == "b"
        assert back["field_c_old"] == "c"
        assert back["extra"] == [1, 2, 3]

    def test_roundtrip_v53_to_v50_and_back(self, chain_migrations):
        original = {
            "config_version": _V53,
            "field_a_new": "a",
            "field_b_new": "b",
            "field_c_new": "c",
        }
        downgraded = migrate_config(original, target_version=_V50)
        back = migrate_config(downgraded, target_version=_V53)
        assert back["config_version"] == _V53
        assert back["field_a_new"] == "a"
        assert back["field_b_new"] == "b"
        assert back["field_c_new"] == "c"


# -----------------------------------------------------------------------------
# Missing intermediate migrations
# -----------------------------------------------------------------------------


class TestMissingIntermediateMigration:
    """Verify that gaps in the migration chain produce clear errors."""

    def test_missing_middle_step_raises_with_context(self, chain_migrations):
        """v50→v53 with v51→v52 removed raises naming the missing step."""
        del versioning._MIGRATIONS[(_V51, _V52)]
        cfg = {"config_version": _V50}
        with pytest.raises(ValueError, match=r"Missing migration from version 51 to 52"):
            migrate_config(cfg, target_version=_V53)

    def test_missing_first_step_raises(self, chain_migrations):
        """v50→v53 with v50→v51 removed raises naming the first step."""
        del versioning._MIGRATIONS[(_V50, _V51)]
        cfg = {"config_version": _V50}
        with pytest.raises(ValueError, match=r"Missing migration from version 50 to 51"):
            migrate_config(cfg, target_version=_V53)

    def test_missing_downgrade_step_raises(self, chain_migrations):
        """v53→v50 with v52→v51 removed raises naming the missing step."""
        del versioning._MIGRATIONS[(_V52, _V51)]
        cfg = {"config_version": _V53}
        with pytest.raises(ValueError, match=r"Missing migration from version 52 to 51"):
            migrate_config(cfg, target_version=_V50)


# -----------------------------------------------------------------------------
# Migration with missing / extra fields
# -----------------------------------------------------------------------------


class TestMissingFields:
    """Migration functions should tolerate configs that lack optional fields."""

    def test_upgrade_with_no_renameable_fields(self, chain_migrations):
        """Config without any of the fields the migrations rename still upgrades."""
        cfg = {"config_version": _V50, "unrelated": "data"}
        result = migrate_config(cfg, target_version=_V53)
        assert result["config_version"] == _V53
        assert result["unrelated"] == "data"
        assert "field_a_new" not in result
        assert "field_b_new" not in result
        assert "field_c_new" not in result

    def test_upgrade_with_partial_fields(self, chain_migrations):
        """Only the fields present are renamed; absent ones are not invented."""
        cfg = {"config_version": _V50, "field_a_old": "a"}
        result = migrate_config(cfg, target_version=_V53)
        assert result["field_a_new"] == "a"
        assert "field_b_new" not in result
        assert "field_c_new" not in result

    def test_downgrade_with_missing_fields(self, chain_migrations):
        """Downgrade with missing fields: absent fields are simply absent."""
        cfg = {"config_version": _V53, "field_c_new": "c"}
        result = migrate_config(cfg, target_version=_V50)
        assert result["config_version"] == _V50
        assert result["field_c_old"] == "c"
        assert "field_a_old" not in result
        assert "field_b_old" not in result

    def test_extra_unknown_fields_are_preserved(self, chain_migrations):
        """Fields not touched by any migration survive the full chain."""
        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "totally_unknown": {"nested": True},
        }
        result = migrate_config(cfg, target_version=_V53)
        assert result["totally_unknown"] == {"nested": True}


# -----------------------------------------------------------------------------
# Error handling mid-migration
# -----------------------------------------------------------------------------


class TestMidMigrationError:
    """If a migration function raises, the error should identify the failing step."""

    def test_error_in_second_step_reports_step(self, chain_migrations):
        """An exception in v51→v52 wraps with the step identifier."""
        del versioning._MIGRATIONS[(_V51, _V52)]

        @register_migration(_V51, _V52)
        def _broken_v51_to_v52(config: dict) -> dict:
            raise RuntimeError("something broke in v51→v52")

        cfg = {"config_version": _V50}
        with pytest.raises(
            RuntimeError,
            match=r"Migration from version 51 to 52 failed.*something broke",
        ):
            migrate_config(cfg, target_version=_V53)

    def test_error_preserves_original_config(self, chain_migrations):
        """Original config is not mutated even when a mid-chain step fails."""
        del versioning._MIGRATIONS[(_V52, _V53)]

        @register_migration(_V52, _V53)
        def _broken_v52_to_v53(config: dict) -> dict:
            raise ValueError("boom")

        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
        }
        original_copy = cfg.copy()
        with pytest.raises(ValueError, match="boom"):
            migrate_config(cfg, target_version=_V53)
        assert cfg == original_copy

    def test_error_preserves_exception_type(self, chain_migrations):
        """The re-raised exception keeps its original type (not generic RuntimeError)."""
        del versioning._MIGRATIONS[(_V51, _V52)]

        @register_migration(_V51, _V52)
        def _type_error_step(config: dict) -> dict:
            raise TypeError("wrong type for field X")

        cfg = {"config_version": _V50}
        with pytest.raises(TypeError, match="Migration from version 51 to 52 failed"):
            migrate_config(cfg, target_version=_V53)


# -----------------------------------------------------------------------------
# register_migration validation
# -----------------------------------------------------------------------------


class TestRegisterMigrationValidation:
    """Edge-case validation for the registration decorator."""

    def test_negative_from_version_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            @register_migration(-1, 0)
            def _bad(config):
                return config

    def test_negative_to_version_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            @register_migration(0, -1)
            def _bad(config):
                return config

    def test_same_version_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            @register_migration(5, 5)
            def _bad(config):
                return config

    def test_duplicate_registration_raises(self):
        """Registering the same (from, to) pair twice raises."""
        @register_migration(_V50, _V51)
        def _first(config):
            return config

        with pytest.raises(ValueError, match="Duplicate migration"):
            @register_migration(_V50, _V51)
            def _second(config):
                return config


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


class TestMigrationLogging:
    """Verify that migration logging reports field changes."""

    def test_upgrade_logs_info_start_and_complete(self, caplog, chain_migrations):
        """INFO logs report migration start (with direction) and completion."""
        cfg = {"config_version": _V50, "field_a_old": "a"}
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V53)
        assert "Migrating config from version 50 to 53 (upgrade)" in caplog.text
        assert "Migration complete: config is now at version 53" in caplog.text

    def test_downgrade_logs_info_direction(self, caplog, chain_migrations):
        """INFO logs report 'downgrade' direction."""
        cfg = {"config_version": _V53, "field_c_new": "c"}
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V50)
        assert "Migrating config from version 53 to 50 (downgrade)" in caplog.text
        assert "Migration complete: config is now at version 50" in caplog.text

    def test_same_version_logs_debug_no_migration(self, caplog, chain_migrations):
        """When source == target, a DEBUG message is emitted and nothing else."""
        cfg = {"config_version": _V50}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V50)
        assert "already at version 50" in caplog.text
        assert "Migrating config" not in caplog.text

    def test_debug_logs_field_rename(self, caplog, chain_migrations):
        """DEBUG logs report removed and added fields for a rename."""
        cfg = {"config_version": _V50, "field_a_old": "a"}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Removed field 'field_a_old'" in caplog.text
        assert "Added field 'field_a_new'" in caplog.text

    def test_debug_logs_updated_field(self, caplog, chain_migrations):
        """DEBUG logs report a field whose value changed in-place."""
        del versioning._MIGRATIONS[(_V50, _V51)]

        @register_migration(_V50, _V51)
        def _mutate_field(config: dict) -> dict:
            config["keep_me"] = config.get("keep_me", 0) + 100
            return config

        cfg = {"config_version": _V50, "keep_me": 1}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Updated field 'keep_me': 1 -> 101" in caplog.text

    def test_debug_logs_no_field_changes(self, caplog, chain_migrations):
        """When migration changes nothing, DEBUG reports 'No field changes'."""
        cfg = {"config_version": _V50}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "No field changes" in caplog.text

    def test_debug_logs_applying_migration_function_name(self, caplog, chain_migrations):
        """DEBUG logs include the qualified name of each migration function."""
        cfg = {"config_version": _V50, "field_a_old": "a"}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Applying migration v50 -> v51" in caplog.text

    def test_multi_step_upgrade_logs_all_steps(self, caplog, chain_migrations):
        """A multi-step upgrade logs each intermediate step."""
        cfg = {
            "config_version": _V50,
            "field_a_old": "a",
            "field_b_old": "b",
            "field_c_old": "c",
        }
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V53)
        assert "Applying migration v50 -> v51" in caplog.text
        assert "Applying migration v51 -> v52" in caplog.text
        assert "Applying migration v52 -> v53" in caplog.text

    def test_info_not_shown_when_no_migration(self, caplog, chain_migrations):
        """When no migration is needed, no INFO messages are emitted."""
        cfg = {"config_version": _V50}
        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V50)
        info_records = [r for r in caplog.records if r.levelno >= logging.INFO]
        assert len(info_records) == 0

    def test_debug_logs_nested_dict_added_key(self, caplog, chain_migrations):
        """DEBUG logs report a key added inside a nested dict."""
        del versioning._MIGRATIONS[(_V50, _V51)]

        @register_migration(_V50, _V51)
        def _add_nested(config: dict) -> dict:
            config.setdefault("video_sets", {})
            config["video_sets"]["new_key"] = "new_value"
            return config

        cfg = {"config_version": _V50, "video_sets": {}}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Added field 'video_sets.new_key' = 'new_value'" in caplog.text

    def test_debug_logs_nested_dict_removed_key(self, caplog, chain_migrations):
        """DEBUG logs report a key removed inside a nested dict."""
        del versioning._MIGRATIONS[(_V50, _V51)]

        @register_migration(_V50, _V51)
        def _remove_nested(config: dict) -> dict:
            config["opts"].pop("old_opt")
            return config

        cfg = {"config_version": _V50, "opts": {"old_opt": 1, "keep": 2}}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Removed field 'opts.old_opt' (was: 1)" in caplog.text

    def test_debug_logs_nested_dict_updated_value(self, caplog, chain_migrations):
        """DEBUG logs report a value changed inside a nested dict."""
        del versioning._MIGRATIONS[(_V50, _V51)]

        @register_migration(_V50, _V51)
        def _update_nested(config: dict) -> dict:
            config["settings"]["threshold"] = 0.9
            return config

        cfg = {"config_version": _V50, "settings": {"threshold": 0.5}}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Updated field 'settings.threshold': 0.5 -> 0.9" in caplog.text

    def test_debug_logs_deeply_nested_change(self, caplog, chain_migrations):
        """DEBUG logs report changes multiple levels deep."""
        del versioning._MIGRATIONS[(_V50, _V51)]

        @register_migration(_V50, _V51)
        def _deep_change(config: dict) -> dict:
            config["a"]["b"]["c"] = "new"
            return config

        cfg = {"config_version": _V50, "a": {"b": {"c": "old"}}}
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            migrate_config(cfg, target_version=_V51)
        assert "Updated field 'a.b.c': 'old' -> 'new'" in caplog.text


# -----------------------------------------------------------------------------
# MigrationMixin compatibility with validate_assignment
# -----------------------------------------------------------------------------


@dataclass(config=ConfigDict(validate_assignment=True))
class ValidatedMigratingConfig(MigrationMixin):
    name: str = "default"
    count: int = 0


class TestMigrationMixinWithValidateAssignment:
    """MigrationMixin must not interfere with validate_assignment (regression)."""

    @pytest.fixture(autouse=True)
    def _default_version(self, monkeypatch):
        monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", 0)

    def test_valid_assignment_takes_effect(self):
        cfg = ValidatedMigratingConfig()
        cfg.count = 42
        assert cfg.count == 42

    def test_invalid_assignment_raises(self):
        cfg = ValidatedMigratingConfig()
        with pytest.raises(ValidationError):
            cfg.count = "not-an-int"
        assert cfg.count == 0

    def test_coercion_applied_on_assignment(self):
        cfg = ValidatedMigratingConfig()
        cfg.count = True  # coerced to 1
        assert cfg.count == 1
