#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Shared fixtures for config tests (no import-time migration registration)."""

import pytest

from deeplabcut.core.config import versioning
from deeplabcut.core.config.versioning import register_migration

_TOY_VERSION_OLD = 98
_TOY_VERSION_NEW = 99
_LEGACY_FIELD = "toy_legacy_field"
_NEW_FIELD = "toy_new_field"


@pytest.fixture(autouse=True)
def isolated_migration_registry(monkeypatch):
    """Snapshot the migration registry per test; production migrations only at baseline."""
    monkeypatch.setattr(versioning, "_MIGRATIONS", versioning._MIGRATIONS.copy())


@pytest.fixture
def register_toy_migrations(monkeypatch):
    """Register v98 <-> v99 toy migrations for the given config_type(s)."""
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", _TOY_VERSION_NEW)

    def register_pair(config_type: str) -> None:
        @register_migration(_TOY_VERSION_OLD, _TOY_VERSION_NEW, config_type=config_type)
        def up(config: dict) -> dict:
            if _LEGACY_FIELD in config:
                config[_NEW_FIELD] = config.pop(_LEGACY_FIELD)
            return config

        @register_migration(_TOY_VERSION_NEW, _TOY_VERSION_OLD, config_type=config_type)
        def down(config: dict) -> dict:
            if _NEW_FIELD in config:
                config[_LEGACY_FIELD] = config.pop(_NEW_FIELD)
            return config

    return register_pair
