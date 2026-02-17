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

import copy
import logging
from typing import Any, Callable, Dict
from typing_extensions import Self
from dataclasses import fields
from functools import wraps

from pydantic import model_validator
from pydantic_core import ArgsKwargs

logger = logging.getLogger(__name__)

# Current configuration schema version
# Increment this when making breaking changes to the config structure
CURRENT_CONFIG_VERSION = 0


# Version registry: maps (from_version, to_version) -> migration function
_MIGRATIONS: Dict[tuple[int, int], Callable[[dict], dict]] = {}


def _diff_dicts(
    before: dict,
    after: dict,
    path: str = "",
    ignore_keys: set[str] | None = None,
) -> list[str]:
    """Recursively diff two dicts and return human-readable change descriptions.

    Args:
        before: The dict before the migration step.
        after: The dict after the migration step.
        path: Dot-separated key path for nested context (used in recursion).
        ignore_keys: Top-level keys to skip (only applied at the root level).

    Returns:
        A list of strings, each describing a single added/removed/updated field.
    """
    ignore = ignore_keys or set()
    before_keys = set(before.keys()) - ignore
    after_keys = set(after.keys()) - ignore

    changes: list[str] = []

    for key in sorted(before_keys - after_keys):
        full = f"{path}.{key}" if path else str(key)
        changes.append(f"Removed field '{full}' (was: {before[key]!r})")

    for key in sorted(after_keys - before_keys):
        full = f"{path}.{key}" if path else str(key)
        changes.append(f"Added field '{full}' = {after[key]!r}")

    for key in sorted(before_keys & after_keys):
        old_val, new_val = before[key], after[key]
        if old_val == new_val:
            continue
        full = f"{path}.{key}" if path else str(key)
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.extend(_diff_dicts(old_val, new_val, path=full))
        else:
            changes.append(f"Updated field '{full}': {old_val!r} -> {new_val!r}")

    return changes


def _log_field_changes(
    before: dict, after: dict, from_v: int, to_v: int
) -> None:
    """Log which fields were added, removed, or changed during a migration step."""
    step = f"v{from_v} -> v{to_v}"
    changes = _diff_dicts(before, after, ignore_keys={"config_version"})

    if changes:
        for change in changes:
            logger.debug("  [%s] %s", step, change)
    else:
        logger.debug("  [%s] No field changes", step)


def register_migration(from_version: int, to_version: int):
    """Decorator to register a migration function.

    Args:
        from_version: The source version number (>= 0).
        to_version: The target version number (>= 0, from_version ± 1 by convention).

    Raises:
        ValueError: If version numbers are invalid or a migration for the same
            (from, to) pair is already registered.

    Example:
        @register_migration(1, 2)
        def migrate_v1_to_v2(config: dict) -> dict:
            # Transform config from version 1 to version 2
            return config
    """
    if from_version < 0 or to_version < 0:
        raise ValueError(
            f"Version numbers must be non-negative, got ({from_version}, {to_version})"
        )
    if from_version == to_version:
        raise ValueError(
            f"from_version and to_version must differ, got ({from_version}, {to_version})"
        )

    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        key = (from_version, to_version)
        if key in _MIGRATIONS:
            raise ValueError(
                f"Duplicate migration registered for {key}. "
                f"Existing: {_MIGRATIONS[key].__wrapped__.__qualname__}, "
                f"new: {func.__qualname__}"
            )

        @wraps(func)
        def wrapper(config: dict) -> dict:
            verbose = logger.isEnabledFor(logging.DEBUG)
            if verbose:
                before = copy.deepcopy(config)
            result = func(config.copy())  # Don't mutate caller's dict
            result["config_version"] = to_version
            if verbose:
                _log_field_changes(before, result, from_version, to_version)
            return result

        _MIGRATIONS[key] = wrapper
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
        logger.debug(
            "Config already at version %d, no migration needed", current_version
        )
        return config
    
    if target_version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            f"Target version {target_version} exceeds current version {CURRENT_CONFIG_VERSION}"
        )
    
    direction = "upgrade" if target_version > current_version else "downgrade"
    logger.info(
        "Migrating config from version %d to %d (%s)",
        current_version, target_version, direction,
    )

    # Chain migrations one step at a time (e.g. 0→1→2 or 3→2→1).
    migrated = config
    step = 1 if target_version > current_version else -1
    for v in range(current_version, target_version, step):
        next_v = v + step
        key = (v, next_v)
        if key not in _MIGRATIONS:
            raise ValueError(
                f"Missing migration from version {v} to {next_v}. "
                f"Available migrations: {list(_MIGRATIONS.keys())}"
            )
        logger.debug(
            "Applying migration v%d -> v%d (%s)",
            v, next_v, _MIGRATIONS[key].__wrapped__.__qualname__,
        )
        try:
            migrated = _MIGRATIONS[key](migrated)
        except Exception as exc:
            raise type(exc)(
                f"Migration from version {v} to {next_v} failed "
                f"({_MIGRATIONS[key].__wrapped__.__qualname__}): {exc}"
            ) from exc

    logger.info("Migration complete: config is now at version %d", target_version)
    return migrated


# ============================================================================
# Migration Definitions
# ============================================================================

@register_migration(0, 1)
def migrate_v0_to_v1(config: dict) -> dict:
    """Migrate from unversioned/legacy config (v0) to v1."""
    # TODO @deruyter92 2026-01-30: Migration logic goes here.
    # e.g. for normalizing field bodyparts from str "MULTI!" to list[str]
    return config


@register_migration(1, 0)
def migrate_v1_to_v0(config: dict) -> dict:
    """Migrate from v1 to v0 (legacy format)."""
    # TODO @deruyter92 2026-01-30: Migration logic goes here.
    return config


# ============================================================================
# MigrationMixin
# ============================================================================


class MigrationMixin:
    """Migration mixin for configuration classes.

    This mixin provides a method to migrate the configuration to the current version
    before validation.
    """
    @model_validator(mode="wrap")
    @classmethod
    def migrate_then_validate(cls, data: Any, handler: Any) -> Self:
        """Run migration before Pydantic's standard model validation.

        Wraps the default validator: migrates raw dict/ArgsKwargs to current
        config version, then delegates to handler for normal validation.
        """
        # If data is already an instance or not a dict, pass through
        if isinstance(data, cls):
            return data

        # Convert to dictionary if ArgsKwargs is passed
        if isinstance(data, ArgsKwargs):
            names = [f.name for f in fields(cls)]
            data = dict(
                zip(names, data.args or []),
                **(data.kwargs or {}),
            )
        if isinstance(data, dict):
            data = migrate_config(
                data, target_version=CURRENT_CONFIG_VERSION
            )
        return handler(data)


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
#     # The wrapper passes a copy, so mutate config directly.
#     # Apply transformations...
#     return config
