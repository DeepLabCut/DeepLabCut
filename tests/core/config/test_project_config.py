#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for ProjectConfig — the real DLCVersionedConfig consumer."""

import logging
from pathlib import Path

import pytest

from deeplabcut.core.config import ProjectConfig, read_config_as_dict
from deeplabcut.core.config.versioning import CURRENT_CONFIG_VERSION

# -----------------------------------------------------------------------------
# Basic construction / defaults
# -----------------------------------------------------------------------------


class TestProjectConfigDefaults:
    def test_defaults(self):
        cfg = ProjectConfig()
        assert cfg.config_version == CURRENT_CONFIG_VERSION
        assert cfg.engine == "pytorch"
        assert cfg.Task == ""
        assert isinstance(cfg.project_path, Path)
        assert cfg.pose_config_path is None
        assert cfg.config_yaml_path == cfg.project_path / "config.yaml"
        assert cfg.config_yaml_path == cfg.project_path / "config.yaml"
        assert cfg.config_yaml_path == cfg.project_path / "config.yaml"

    def test_from_dict_minimal(self):
        cfg = ProjectConfig.from_dict(
            {
                "Task": "mytask",
                "scorer": "scorer",
                "date": "Jan01",
                "project_path": "/tmp/project",
            }
        )
        assert cfg.Task == "mytask"
        assert cfg.project_path == Path("/tmp/project")

    def test_from_dict_rejects_unknown_keys(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProjectConfig.from_dict({"Task": "t", "not_a_real_field": 1})

    def test_is_versioned_config(self):
        cfg = ProjectConfig()
        assert hasattr(cfg, "is_dirty")
        assert not cfg.is_dirty


# -----------------------------------------------------------------------------
# bodyparts_list
# -----------------------------------------------------------------------------


class TestProjectConfigBodypartsList:
    def test_single_animal_bodyparts_list_returns_bodyparts(self):
        cfg = ProjectConfig(multianimalproject=False, bodyparts=["nose", "tail"])
        assert cfg.bodyparts_list == ["nose", "tail"]

    def test_multi_animal_bodyparts_list_returns_multianimalbodyparts(self):
        cfg = ProjectConfig(
            multianimalproject=True,
            bodyparts="MULTI!",
            multianimalbodyparts=["nose", "tail"],
        )
        assert cfg.bodyparts_list == ["nose", "tail"]

    def test_single_animal_multi_sentinel_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="MULTI!"):
            ProjectConfig(multianimalproject=False, bodyparts="MULTI!")


# -----------------------------------------------------------------------------
# validate_project_path / from_any(repair_path=True)
# -----------------------------------------------------------------------------


def _write_wrong_project_path_yaml(config_path: Path) -> None:
    config_path.write_text("project_path: /completely/wrong/path\nengine: pytorch\n")


class TestValidateProjectPath:
    def test_raises_when_yaml_not_found(self, tmp_path):
        cfg = ProjectConfig(project_path=tmp_path)
        with pytest.raises(FileNotFoundError, match="config.yaml not found"):
            cfg.validate_project_path()

    @pytest.mark.parametrize("write", [False, True])
    def test_validate_project_path_write_flag(self, tmp_path, write):
        config_path = tmp_path / "config.yaml"
        _write_wrong_project_path_yaml(config_path)
        cfg = ProjectConfig(project_path=Path("/completely/wrong/path"))
        cfg.validate_project_path(yaml_path=config_path, write=write)
        assert cfg.project_path == tmp_path
        saved = read_config_as_dict(config_path)
        if write:
            assert Path(saved["project_path"]) == tmp_path
            assert not cfg.is_dirty
        else:
            assert saved["project_path"] == "/completely/wrong/path"
            assert "project_path" in cfg.dirty_fields

    @pytest.mark.parametrize(
        ("config_input", "repair_path"),
        [
            ("yaml_path", True),
            ("dict", True),
            ("dict", False),
        ],
        ids=["yaml_path", "dict_repair", "dict_no_repair"],
    )
    def test_from_any_repair_path(self, tmp_path, config_input, repair_path):
        config_path = tmp_path / "config.yaml"
        _write_wrong_project_path_yaml(config_path)
        if config_input == "yaml_path":
            cfg = ProjectConfig.from_any(config_path, repair_path=repair_path)
        else:
            cfg = ProjectConfig.from_any(
                {"project_path": tmp_path, "engine": "pytorch"},
                repair_path=repair_path,
            )
        assert cfg.project_path == tmp_path
        assert not cfg.is_dirty
        saved = read_config_as_dict(config_path)
        if config_input == "yaml_path" and repair_path:
            assert Path(saved["project_path"]) == tmp_path
        else:
            assert saved["project_path"] == "/completely/wrong/path"


# -----------------------------------------------------------------------------
# _post_yaml_load_updates: project_path repair
# -----------------------------------------------------------------------------


class TestPostYamlLoadUpdates:
    def test_project_path_updated_when_differs_from_yaml_location(self, tmp_path):
        """When the stored project_path differs from the YAML's directory,
        _post_yaml_load_updates corrects it and marks the field dirty."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("project_path: /completely/wrong/path\nengine: pytorch\n")
        cfg = ProjectConfig.from_yaml(config_path)
        assert cfg.project_path == tmp_path
        assert "project_path" in cfg.dirty_fields
        assert any("project_path updated" in n for n in cfg.change_notes)

    def test_project_path_not_updated_when_already_correct(self, tmp_path):
        """When the stored project_path already matches the YAML directory,
        nothing is changed and the config stays clean."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"project_path: {tmp_path}\nengine: pytorch\n")
        cfg = ProjectConfig.from_yaml(config_path)
        assert cfg.project_path == tmp_path
        assert not cfg.is_dirty


# -----------------------------------------------------------------------------
# Legacy empty / null values (normalize_legacy_empty_values)
# -----------------------------------------------------------------------------


class TestNormalizeLegacyEmptyValues:
    def test_none_for_non_optional_field_uses_default(self):
        """Bare/null YAML values become None in dicts; non-optional fields use defaults."""
        cfg = ProjectConfig.from_dict({"multianimalproject": None, "engine": "pytorch"})
        assert cfg.multianimalproject is False

    def test_none_for_optional_field_kept_as_none(self):
        """Optional fields (T | None) accept None; it is not rewritten to a non-None default."""
        cfg = ProjectConfig.from_dict({"identity": None, "engine": "pytorch"})
        assert cfg.identity is None

    def test_empty_string_for_legacy_list_fields_uses_default(self):
        cfg = ProjectConfig.from_dict(
            {
                "skeleton": "",
                "TrainingFraction": "",
                "video_sets": "",
                "bodyparts": "",
                "engine": "pytorch",
            }
        )
        defaults = ProjectConfig()
        assert cfg.skeleton == defaults.skeleton
        assert cfg.TrainingFraction == defaults.TrainingFraction
        assert cfg.video_sets == defaults.video_sets
        assert cfg.bodyparts == defaults.bodyparts

    def test_from_dict_and_from_yaml_agree_on_null_keys(self, tmp_path):
        """Dict path (legacy scripts) must match YAML path for bare/null keys."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    f"project_path: {tmp_path}",
                    "engine: pytorch",
                    "multianimalproject:",
                    "identity:",
                    "bodyparts:",
                    "  - snout",
                    "",
                ]
            )
        )
        from_yaml = ProjectConfig.from_yaml(config_path)
        from_dict = ProjectConfig.from_dict(read_config_as_dict(config_path))
        assert from_dict.multianimalproject == from_yaml.multianimalproject is False
        assert from_dict.identity == from_yaml.identity is None
        assert from_dict.bodyparts == from_yaml.bodyparts == ["snout"]

    def test_openfield_example_loads_via_dict_and_yaml(self):
        """Example project configs with empty fields must load via both entrypoints."""
        config_path = Path(__file__).resolve().parents[3] / "examples" / "openfield-Pranav-2018-10-30" / "config.yaml"
        assert config_path.is_file()
        raw = read_config_as_dict(config_path)
        assert raw["multianimalproject"] is None
        assert raw["identity"] is None

        from_dict = ProjectConfig.from_dict(raw)
        from_yaml = ProjectConfig.from_yaml(config_path)
        assert from_dict.multianimalproject == from_yaml.multianimalproject is False
        assert from_dict.identity == from_yaml.identity is None

    def test_null_non_optional_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="deeplabcut.core.config.project_config"):
            ProjectConfig.from_dict({"multianimalproject": None, "engine": "pytorch"})
        assert any("multianimalproject" in r.message for r in caplog.records)


# -----------------------------------------------------------------------------
# YAML round-trip
# -----------------------------------------------------------------------------


class TestProjectConfigYamlRoundtrip:
    def test_roundtrip_preserves_scalar_fields(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        orig = ProjectConfig(
            Task="pose",
            scorer="scorer1",
            date="2026-01-01",
            project_path=tmp_path,
            engine="pytorch",
            pcutoff=0.6,
        )
        orig.to_yaml(config_path)
        loaded = ProjectConfig.from_yaml(config_path)
        assert loaded.Task == "pose"
        assert loaded.scorer == "scorer1"
        assert loaded.pcutoff == 0.6
        assert loaded.engine == "pytorch"

    def test_roundtrip_config_is_clean(self, tmp_path):
        """After saving and reloading, the loaded config must be clean
        (project_path already matches YAML location)."""
        config_path = tmp_path / "config.yaml"
        orig = ProjectConfig(project_path=tmp_path)
        orig.to_yaml(config_path)
        loaded = ProjectConfig.from_yaml(config_path)
        assert not loaded.is_dirty

    def test_roundtrip_no_unknown_keys_in_yaml(self, tmp_path):
        """The saved YAML must not contain keys that ProjectConfig doesn't declare."""
        config_path = tmp_path / "config.yaml"
        ProjectConfig(project_path=tmp_path).to_yaml(config_path)
        saved = read_config_as_dict(config_path)
        declared = set(ProjectConfig.model_fields.keys())
        unknown = set(saved.keys()) - declared
        assert unknown == set(), f"Unknown keys written to YAML: {unknown}"

    def test_to_yaml_writes_config_version(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        ProjectConfig(project_path=tmp_path).to_yaml(config_path)
        saved = read_config_as_dict(config_path)
        assert "config_version" in saved
        assert saved["config_version"] == CURRENT_CONFIG_VERSION


# -----------------------------------------------------------------------------
# Migration / versioning integration
# -----------------------------------------------------------------------------


class TestProjectConfigVersioning:
    def test_no_migration_log_for_current_version_yaml(self, tmp_path, caplog):
        """Loading a config already at CURRENT_CONFIG_VERSION emits no migration INFO log."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"config_version: {CURRENT_CONFIG_VERSION}\nproject_path: {tmp_path}\nengine: pytorch\n")
        with caplog.at_level(logging.INFO, logger="deeplabcut.core.config.versioning"):
            ProjectConfig.from_yaml(config_path)

        migration_records = [r for r in caplog.records if "igrating" in r.message]
        assert len(migration_records) == 0

    def test_unversioned_yaml_treated_as_v0(self, tmp_path):
        """A YAML file without config_version is treated as version 0."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"project_path: {tmp_path}\nengine: pytorch\n")
        cfg = ProjectConfig.from_yaml(config_path)
        assert cfg.config_version == CURRENT_CONFIG_VERSION


# -----------------------------------------------------------------------------
# Fields awaiting the next schema version bump
#
# Each test here describes a field that is exposed as a read-only property today
# because promoting it to a real schema field requires bumping
# CURRENT_CONFIG_VERSION and writing the migration. They are marked strict xfail
# so that the migration landing turns them into failures, as a reminder to drop
# the property and the marker together.
# -----------------------------------------------------------------------------


class TestFieldsPendingSchemaMigration:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "bboxes_pcutoff is a property until the schema version is bumped, "
            "see https://github.com/DeepLabCut/DeepLabCut/pull/3470"
        ),
    )
    def test_bboxes_pcutoff_is_a_schema_field(self):
        assert "bboxes_pcutoff" in ProjectConfig.model_fields
