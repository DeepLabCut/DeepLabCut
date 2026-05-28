#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for change tracking on DLCVersionedConfig."""

from __future__ import annotations

import logging

import pytest
from pydantic import Field, ValidationError

from deeplabcut.core.config import DLCVersionedConfig
from deeplabcut.utils.deprecation import DLCDeprecationWarning


class TrackedConfig(DLCVersionedConfig):
    name: str = "default"
    count: int = 0
    flag: bool = False


# ------------------------------------------------------------------
# Dirty-state tracking
# ------------------------------------------------------------------


class TestDirtyState:
    def test_clean_after_construction(self):
        cfg = TrackedConfig()
        assert not cfg.is_dirty
        assert cfg.dirty_fields == frozenset()

    def test_dirty_after_field_change(self):
        cfg = TrackedConfig()
        cfg.name = "changed"
        assert cfg.is_dirty
        assert "name" in cfg.dirty_fields

    def test_not_dirty_when_set_to_same_value(self):
        cfg = TrackedConfig(name="same")
        cfg.name = "same"
        assert not cfg.is_dirty

    def test_multiple_fields_tracked(self):
        cfg = TrackedConfig()
        cfg.name = "new"
        cfg.count = 5
        assert cfg.dirty_fields == frozenset({"name", "count"})

    def test_dirty_fields_is_frozen(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        fs = cfg.dirty_fields
        assert isinstance(fs, frozenset)


# ------------------------------------------------------------------
# mark_clean
# ------------------------------------------------------------------


class TestMarkClean:
    def test_mark_clean_resets_dirty_fields(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.mark_clean()
        assert not cfg.is_dirty
        assert cfg.dirty_fields == frozenset()

    def test_mark_clean_resets_change_notes(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.record_change_note("name", "renamed")
        cfg.mark_clean()
        assert cfg.change_notes == []

    def test_dirty_again_after_mark_clean(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.mark_clean()
        cfg.count = 10
        assert cfg.is_dirty
        assert cfg.dirty_fields == frozenset({"count"})


# ------------------------------------------------------------------
# Change notes
# ------------------------------------------------------------------


class TestChangeNotes:
    def test_no_notes_by_default(self):
        cfg = TrackedConfig()
        assert cfg.change_notes == []

    def test_record_and_retrieve_note(self):
        cfg = TrackedConfig()
        cfg.name = "new"
        cfg.record_change_note("name", "name was updated to 'new'")
        assert cfg.change_notes == ["name was updated to 'new'"]

    def test_note_overwrites_previous_for_same_field(self):
        cfg = TrackedConfig()
        cfg.name = "a"
        cfg.record_change_note("name", "first")
        cfg.record_change_note("name", "second")
        assert cfg.change_notes == ["second"]

    def test_multiple_notes_for_different_fields(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.count = 1
        cfg.record_change_note("name", "note-name")
        cfg.record_change_note("count", "note-count")
        assert set(cfg.change_notes) == {"note-name", "note-count"}

    def test_include_caller_appends_tag(self):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.record_change_note("name", "updated", include_caller=True)
        notes = cfg.change_notes
        assert len(notes) == 1
        assert notes[0].startswith("updated [")
        assert "test_change_tracking.py:" in notes[0]


# ------------------------------------------------------------------
# log_changes
# ------------------------------------------------------------------


class TestLogChanges:
    def test_log_changes_no_output_when_clean(self, caplog):
        cfg = TrackedConfig()
        with caplog.at_level(logging.INFO):
            cfg.log_changes()
        assert caplog.text == ""

    def test_log_changes_includes_note(self, caplog):
        cfg = TrackedConfig()
        cfg.name = "x"
        cfg.record_change_note("name", "name was updated")
        with caplog.at_level(logging.INFO):
            cfg.log_changes()
        assert "name was updated" in caplog.text

    def test_log_changes_shows_field_without_note(self, caplog):
        cfg = TrackedConfig()
        cfg.count = 99
        with caplog.at_level(logging.INFO):
            cfg.log_changes()
        assert "count was modified" in caplog.text

    def test_log_changes_mixes_notes_and_bare_fields(self, caplog):
        cfg = TrackedConfig()
        cfg.name = "y"
        cfg.count = 5
        cfg.record_change_note("name", "renamed to y")
        with caplog.at_level(logging.INFO):
            cfg.log_changes()
        assert "renamed to y" in caplog.text
        assert "count was modified" in caplog.text

    def test_log_changes_header_contains_class_name(self, caplog):
        cfg = TrackedConfig()
        cfg.flag = True
        with caplog.at_level(logging.INFO):
            cfg.log_changes()
        assert "TrackedConfig" in caplog.text


# ------------------------------------------------------------------
# Integration with DLCVersionedConfig.from_yaml
# ------------------------------------------------------------------


class TestFromYamlIntegration:
    def test_clean_after_from_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("name: loaded\ncount: 42\nflag: true\n")
        cfg = TrackedConfig.from_yaml(path)
        assert cfg.name == "loaded"
        assert not cfg.is_dirty
        assert cfg.change_notes == []

    def test_roundtrip_stays_clean(self, tmp_path):
        path = tmp_path / "config.yaml"
        orig = TrackedConfig(name="rt", count=7)
        orig.to_yaml(path)
        loaded = TrackedConfig.from_yaml(path)
        assert not loaded.is_dirty


# ------------------------------------------------------------------
# Interaction with pydantic validate_assignment
# ------------------------------------------------------------------


class TestValidateAssignment:
    """Verify that change tracking wraps *around* pydantic's validator."""

    def test_valid_assignment_is_tracked(self):
        cfg = TrackedConfig()
        cfg.count = 7
        assert cfg.count == 7
        assert "count" in cfg.dirty_fields

    def test_invalid_assignment_rejected_and_not_tracked(self):
        cfg = TrackedConfig()
        with pytest.raises(ValidationError):
            cfg.count = "not-an-int"
        assert cfg.count == 0
        assert not cfg.is_dirty

    def test_coerced_value_stored(self):
        """Pydantic coerces compatible types (e.g. bool -> int); the coerced
        value should be stored and tracked."""
        cfg = TrackedConfig()
        cfg.count = True  # coerced to 1
        assert cfg.count == 1
        assert "count" in cfg.dirty_fields


# ------------------------------------------------------------------
# Instance isolation — two instances of the same class are independent
# ------------------------------------------------------------------


class TestInstanceIsolation:
    """The class-level __setattr__ patch must not cause instances to share state."""

    def test_two_instances_independent_dirty_sets(self):
        cfg1 = TrackedConfig()
        cfg2 = TrackedConfig()
        cfg1.name = "one"
        assert "name" in cfg1.dirty_fields
        assert not cfg2.is_dirty

    def test_mark_clean_on_one_does_not_affect_other(self):
        cfg1 = TrackedConfig()
        cfg2 = TrackedConfig()
        cfg1.name = "x"
        cfg2.count = 5
        cfg1.mark_clean()
        assert not cfg1.is_dirty
        assert cfg2.is_dirty
        assert "count" in cfg2.dirty_fields

    def test_different_subclasses_independent(self):
        class ConfigA(DLCVersionedConfig):
            x: int = 0

        class ConfigB(DLCVersionedConfig):
            y: int = 0

        a = ConfigA()
        b = ConfigB()
        a.x = 1
        assert "x" in a.dirty_fields
        assert not b.is_dirty


# ------------------------------------------------------------------
# from_dict and to_yaml lifecycle
# ------------------------------------------------------------------


class TestLifecycle:
    def test_from_dict_is_clean(self):
        cfg = TrackedConfig.from_dict({"name": "loaded", "count": 5})
        assert not cfg.is_dirty
        assert cfg.change_notes == []

    def test_to_yaml_marks_writer_clean(self, tmp_path):
        path = tmp_path / "out.yaml"
        cfg = TrackedConfig(name="dirty")
        cfg.name = "changed"
        assert cfg.is_dirty
        cfg.to_yaml(path)
        assert not cfg.is_dirty

    def test_to_yaml_with_mark_clean_false_does_not_reset(self, tmp_path):
        path = tmp_path / "out.yaml"
        cfg = TrackedConfig()
        cfg.name = "changed"
        cfg.to_yaml(path, mark_clean=False)
        assert cfg.is_dirty

    def test_record_change_note_for_unmodified_field_is_stored(self):
        """record_change_note does not require the field to be dirty; it just stores the note."""
        cfg = TrackedConfig()
        cfg.record_change_note("name", "set during init")
        assert cfg.change_notes == ["set during init"]
        assert not cfg.is_dirty  # still clean — no assignment was made


# ------------------------------------------------------------------
# Alias access with change tracking
# ------------------------------------------------------------------


class TrackedAliasConfig(DLCVersionedConfig):
    project_path: str = Field(
        default="",
        json_schema_extra={"aliases": ["projectPath"]},
    )


class TestAliasChangeTracking:
    def test_setitem_alias_marks_canonical_field_dirty(self):
        cfg = TrackedAliasConfig()
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            cfg["projectPath"] = "/new"
        assert cfg.project_path == "/new"
        assert "project_path" in cfg.dirty_fields

    def test_setattr_alias_marks_canonical_field_dirty(self):
        cfg = TrackedAliasConfig()
        with pytest.warns(DLCDeprecationWarning, match="projectPath"):
            cfg.projectPath = "/attr"
        assert cfg.project_path == "/attr"
        assert "project_path" in cfg.dirty_fields
