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
"""Configuration migration system for handling version upgrades and downgrades.

This module provides a versioned migration system that allows configurations
to be upgraded from older versions to newer ones, or downgraded to older formats.
Upgrade migrations are chained together, so any version can be upgraded to the
latest by applying all intermediate migrations in sequence. Downgrade migrations
can be registered for specific version pairs when backward compatibility is needed.
"""

from typing import Callable, Dict
from functools import wraps


# Current configuration schema version
# Increment this when making breaking changes to the config structure
CURRENT_CONFIG_VERSION = 0


# Version registry: maps (from_version, to_version) -> migration function
_MIGRATIONS: Dict[tuple[int, int], Callable[[dict], dict]] = {}


def register_migration(from_version: int, to_version: int):
    """Decorator to register a migration function.
    
    Args:
        from_version: The source version number
        to_version: The target version number (must be from_version + 1)
    
    Example:
        @register_migration(1, 2)
        def migrate_v1_to_v2(config: dict) -> dict:
            # Transform config from version 1 to version 2
            return config
    """
    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _MIGRATIONS[(from_version, to_version)] = func
        
        @wraps(func)
        def wrapper(config: dict) -> dict:
            result = func(config.copy())  # Don't mutate input
            result["config_version"] = to_version
            return result
        
        return wrapper
    return decorator


def get_config_version(config: dict) -> int:
    """Extract the configuration version from a config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Version number (0 for legacy/unversioned configs)
    """
    return config.get("config_version", 0)


def migrate_config(config: dict, target_version: int = CURRENT_CONFIG_VERSION) -> dict:
    """Migrate a configuration to the target version.
    
    Applies all necessary migrations in sequence to upgrade the config
    from its current version to the target version.
    
    Args:
        config: Configuration dictionary to migrate
        target_version: Target version to migrate to (default: current)
        
    Returns:
        Migrated configuration dictionary
        
    Raises:
        ValueError: If migration chain is incomplete or target version is invalid
    """
    current_version = get_config_version(config)
    
    if current_version == target_version:
        return config
    
    if target_version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            f"Target version {target_version} exceeds current version {CURRENT_CONFIG_VERSION}"
        )
    
    # Apply migrations sequentially
    migrated = config.copy()
    migration_key = (current_version, target_version)
    if migration_key not in _MIGRATIONS:
        raise ValueError(
            f"Missing migration from version {current_version} to {target_version}. "
            f"Available migrations: {list(_MIGRATIONS.keys())}"
        )
    
    migration_func = _MIGRATIONS[migration_key]
    migrated = migration_func(migrated)
    
    return migrated


# ============================================================================
# Migration Definitions
# ============================================================================

@register_migration(0, 1)
def migrate_v0_to_v1(config: dict) -> dict:
    """Migrate from unversioned/legacy config (v0) to v1.
    """
    normalized = config.copy()
    
    # TODO @deruyter92 2026-01-30: Migration logic goes here.
    # e.g. for normalizing field bodyparts from str "MULTI!" to list[str]
    
    return normalized


@register_migration(1, 0)
def migrate_v1_to_v0(config: dict) -> dict:
    """Migrate from v1 to v0 (legacy format).
    """
    normalized = config.copy()

    # TODO @deruyter92 2026-01-30: Migration logic goes here.

    return normalized


# ============================================================================
# Future migrations can be added here following the same pattern:
# ============================================================================
#
# @register_migration(2, 3)
# def migrate_v2_to_v3(config: dict) -> dict:
#     """Migrate from v2 to v3.
#     
#     Describe what changes in this version.
#     """
#     normalized = config.copy()
#     # Apply transformations...
#     return normalized
