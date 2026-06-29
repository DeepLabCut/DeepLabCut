"""Pathological casess tests for the centralized config model."""

from __future__ import annotations

import enum
import logging
from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from deeplabcut.core.config import DLCBaseConfig, DLCVersionedConfig, ProjectConfig, versioning
from deeplabcut.core.deprecation import DLCDeprecationWarning

# -----------------------------------------------------------------------------
# In-place nested mutation
# -----------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="This may make validation difficult and cause subtle issues "
    "if we start changing how functions such as append() operate."
    " Avoiding lists in configs could be a better long-term solution."
)
def test_in_place_list_mutation_should_be_validated_like_assignment():
    """Appending to a config list changes config state without calling __setattr__.

    A strict config system should reject invalid values even when users mutate an
    existing list instead of assigning a replacement list.

    Note that this pattern would incur validation:
        cfg.TrainingFraction = [*cfg.TrainingFraction, 0.8]
        cfg.video_sets = {**cfg.video_sets, "video.mp4": {"crop": "0, 100, 0, 100"}}
    """
    cfg = ProjectConfig(TrainingFraction=[0.95])
    cfg.mark_clean()

    with pytest.raises(ValidationError):
        cfg.TrainingFraction.append(1.5)


@pytest.mark.xfail(reason="Not implemented yet")
@pytest.mark.parametrize(
    ("field_name", "mutate"),
    [
        ("TrainingFraction", lambda cfg: cfg.TrainingFraction.append(0.8)),
        ("video_sets", lambda cfg: cfg.video_sets.__setitem__("video.mp4", {"crop": "0, 100, 0, 100"})),
    ],
)
def test_in_place_nested_mutation_should_mark_config_dirty(field_name, mutate):
    """Mutating nested containers changes the saved config but may bypass dirty tracking.

    Dirty tracking is useful only if semantic config changes are recorded, whether
    the user assigns a whole field or mutates a nested list/dict in place.
    This ties with the problem that some nested fields are still dict instead of config models,
    maybe once we have full config models we can come up with a parent/child system where changes in a child
    are detected when querying the parent for its dirty state (or a simpler equivalent)
    """
    cfg = ProjectConfig(TrainingFraction=[0.95], video_sets={})
    cfg.mark_clean()

    mutate(cfg)

    assert cfg.is_dirty
    assert field_name in cfg.dirty_fields


# -----------------------------------------------------------------------------
# Private state isolation
# -----------------------------------------------------------------------------


class _TrackedForIsolation(DLCVersionedConfig):
    name: str = "default"
    count: int = 0


def test_dirty_state_is_not_shared_between_instances():
    """Private dirty-field state must be per-instance, not shared via a mutable default."""
    first = _TrackedForIsolation()
    second = _TrackedForIsolation()

    first.name = "changed"

    assert "name" in first.dirty_fields
    assert not second.is_dirty
    assert "name" not in second.dirty_fields


def test_change_notes_are_not_shared_between_instances():
    """Private change-note state must be per-instance, not shared via a mutable default."""
    first = _TrackedForIsolation()
    second = _TrackedForIsolation()

    first.record_change_note("name", "name changed on first instance")

    assert first.change_notes == ["name changed on first instance"]
    assert second.change_notes == []


# -----------------------------------------------------------------------------
# Changes note consistency
# -----------------------------------------------------------------------------


def test_change_note_should_reject_unknown_field_names():
    """Notes for misspelled fields are silently lost during logging.

    Rejecting unknown field names catches typos at the call site instead of
    storing a note that can never match a dirty field.
    """
    cfg = ProjectConfig()

    with pytest.raises(KeyError):
        cfg.record_change_note("not_a_real_field", "this note should not be accepted")


def test_change_note_recorded_with_alias_should_follow_canonical_dirty_field(caplog):
    """Aliases should not split notes from the canonical dirty field name.

    If a note is recorded with a deprecated alias, log_changes should still use
    that note when the canonical field is modified.
    """
    cfg = ProjectConfig()
    cfg.mark_clean()

    with pytest.warns(DLCDeprecationWarning, match="with_identity"):
        cfg.record_change_note("with_identity", "identity changed through compatibility alias")
    cfg.identity = True

    with caplog.at_level(logging.INFO):
        cfg.log_changes()

    assert "identity changed through compatibility alias" in caplog.text


# -----------------------------------------------------------------------------
# Nested YAML comments and nested serialization
# -----------------------------------------------------------------------------


class _CommentedInner(DLCBaseConfig):
    threshold: float = Field(
        default=0.5,
        json_schema_extra={"comment": "Nested threshold comment"},
    )


class _CommentedOuter(DLCBaseConfig):
    inner: _CommentedInner = Field(
        default_factory=_CommentedInner,
        json_schema_extra={"comment": "Inner config section"},
    )


def test_to_yaml_should_emit_comments_for_nested_config_fields(tmp_path):
    """Nested config models can have field comments too.

    This protects against only applying YAML comments to the top-level model and
    silently dropping useful nested schema documentation.
    """
    path = tmp_path / "commented.yaml"

    _CommentedOuter().to_yaml(path)

    text = path.read_text()
    assert "Inner config section" in text
    assert "Nested threshold comment" in text


class _SerializationMode(enum.Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class _SerializableInner(DLCBaseConfig):
    output_path: Path = Path("outputs/predictions")
    mode: _SerializationMode = _SerializationMode.FAST


class _SerializableOuter(DLCBaseConfig):
    inner: _SerializableInner = Field(default_factory=_SerializableInner)


def test_nested_config_normalization_should_handle_paths_and_enums():
    """Nested model dumps should normalize non-primitive values before YAML output.

    Path and Enum values are common in configs and should become plain YAML-safe
    values even when they appear inside nested config models.
    """
    cfg = _SerializableOuter()

    assert cfg.to_dict(normalize=True) == {
        "inner": {
            "output_path": str(Path("outputs/predictions")),
            "mode": "fast",
        }
    }


def test_nested_config_to_yaml_should_write_plain_path_and_enum_values(tmp_path):
    """YAML output should not rely on Python-specific object representation.

    This catches failures where nested BaseModel values, pathlib paths, or enums
    are passed to the YAML dumper without normalization.
    """
    path = tmp_path / "serializable.yaml"

    _SerializableOuter().to_yaml(path)

    text = path.read_text()
    assert str(Path("outputs/predictions")) in text
    assert "fast" in text


# -----------------------------------------------------------------------------
# Alias warnings and canonical writes
# -----------------------------------------------------------------------------


def test_item_assignment_with_alias_should_warn_once_and_track_canonical_field():
    """Dict-style alias assignment should not warn twice or track the alias name.

    __setitem__ resolves aliases before delegating to assignment validation, so
    the visible side effects should be one warning and one canonical dirty field.
    """
    cfg = ProjectConfig()
    cfg.mark_clean()

    with pytest.warns(DLCDeprecationWarning, match="with_identity") as caught:
        cfg["with_identity"] = True

    assert len(caught) == 1
    assert cfg.identity is True
    assert "identity" in cfg.dirty_fields
    assert "with_identity" not in cfg.dirty_fields


# -----------------------------------------------------------------------------
# Cross-field validation vs bulk update
# -----------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "update() applies overrides one setattr at a time; cross-field "
        "model validators run on each assignment, so intermediate states "
        "like multianimalproject=True with bodyparts=[] are rejected."
    ),
)
def test_update_applies_cross_field_overrides_atomically():
    """Bulk update should validate the merged state, not each setattr in isolation."""
    cfg = ProjectConfig(bodyparts=[], multianimalproject=False)
    # Invalid state, midway updating: when bodyparts is set to "MULTI!" before multianimalproject is set to True.
    cfg.update({"bodyparts": "MULTI!", "multianimalproject": True})
    assert cfg.multianimalproject is True
    assert cfg.bodyparts == "MULTI!"


# -----------------------------------------------------------------------------
# Migration skipped on typed construction
# -----------------------------------------------------------------------------


class _TypedConstructionCfg(DLCVersionedConfig):
    config_version: int = 1
    epochs: int = 50


@pytest.mark.parametrize("kwargs", [{}, {"epochs": 100}])
def test_migration_skipped_on_typed_construction(monkeypatch, kwargs):
    """Typed construction must not run the legacy YAML migration chain.

    ``MyCfg(epochs=100)`` is current-schema construction, not loading an old file.
    Migration should only run via ``from_dict`` / ``from_yaml``.
    """
    monkeypatch.setattr(versioning, "CURRENT_CONFIG_VERSION", 1)

    cfg = _TypedConstructionCfg(**kwargs)

    assert cfg.epochs == kwargs.get("epochs", 50)
    assert cfg.config_version == 1
