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
"""Tests capturing the current state of the generalized_data_converter configuration.

These tests document default values and behavior so that a migration to a new
configuration system can preserve existing behavior. Add or adjust assertions
if you intentionally change defaults.
"""

import pytest

from deeplabcut.core.config import read_config_as_dict, write_config
from deeplabcut.modelzoo.generalized_data_converter.datasets.materialize import (
    MaDLC_config,
    SingleDLC_config,
    modify_train_test_cfg,
)

# Modelzoo SingleDLC_config defaults
# 2026-01-29 config version 0, before centralized typed configs were added
SINGLE_DLC_DEFAULTS = {
    "Task": "",
    "project_path": "",
    "scorer": "",
    "date": "",
    "video_sets": "",
    "skeleton": "",
    "bodyparts": [],
    "start": 0,
    "stop": 1,
    "numframes2pick": 42,
    "skeleton_color": "black",
    "pcutoff": 0.6,
    "dotsize": 8,
    "alphavalue": 0.7,
    "colormap": "rainbow",
    "TrainingFraction": "",
    "iteration": 0,
    "default_net_type": "resnet_50",
    "default_augmenter": "imgaug",
    "snapshotindex": -1,
    "batch_size": 8,
    "cropping": False,
    "croppedtraining": False,
    "multianimalproject": False,
    "uniquebodyparts": [],
    "x1": 0,
    "x2": 640,
    "y1": 277,
    "y2": 624,
    "corner2move2": [50, 50],
    "move2corner": True,
    "identity": False,
}


# MaDLC_config defaults; only differences from Single are listed in tests
MA_DLC_DEFAULTS_OVERRIDES = {
    "default_augmenter": "multi-animal-imgaug",
    "bodyparts": "MULTI!",
    "croppedtraining": True,
    "multianimalproject": True,
}
MA_DLC_EXTRA_KEYS = {"individuals", "multianimalbodyparts"}


class TestModelzooProjectConfigDefaults:
    def test_single_dlc_config_defaults(self):
        config = SingleDLC_config()
        assert set(config.cfg.keys()) == set(SINGLE_DLC_DEFAULTS.keys())
        for key, expected in SINGLE_DLC_DEFAULTS.items():
            assert config.cfg[key] == expected, f"SingleDLC_config.cfg[{key!r}]"

    def test_ma_dlc_config_has_all_single_keys_plus_ma_specific(self):
        single_keys = set(SINGLE_DLC_DEFAULTS.keys())
        ma_config = MaDLC_config()
        ma_keys = set(ma_config.cfg.keys())
        assert single_keys <= ma_keys
        assert MA_DLC_EXTRA_KEYS <= ma_keys

    def test_ma_dlc_config_overrides_vs_single(self):
        single_config = SingleDLC_config()
        ma_config = MaDLC_config()
        for key, expected in MA_DLC_DEFAULTS_OVERRIDES.items():
            assert ma_config.cfg[key] == expected, f"MaDLC_config.cfg[{key!r}]"
            assert ma_config.cfg[key] != single_config.cfg[key], f"MaDLC should differ from Single for {key!r}"

    def test_ma_dlc_config_shared_defaults_match_single(self):
        """Keys not overridden in Ma should match Single defaults."""
        single_config = SingleDLC_config()
        ma_config = MaDLC_config()
        for key in SINGLE_DLC_DEFAULTS:
            if key in MA_DLC_DEFAULTS_OVERRIDES:
                continue
            assert ma_config.cfg[key] == single_config.cfg[key], f"MaDLC_config.cfg[{key!r}] should match Single"


class TestModelzooCreateProjectConfig:
    """Behavior of create_cfg: file location, format, and update semantics."""

    def test_single_dlc_create_cfg_writes_config_yaml(self, tmp_path):
        config = SingleDLC_config()
        config.create_cfg(tmp_path, {"Task": "mytask"})
        path = tmp_path / "config.yaml"
        assert path.exists()
        data = read_config_as_dict(path)
        assert data["Task"] == "mytask"
        assert data["default_net_type"] == "resnet_50"

    def test_create_cfg_overwrites_with_kwargs(self, tmp_path):
        config = SingleDLC_config()
        config.create_cfg(tmp_path, {"Task": "mytask", "batch_size": 16})
        path = tmp_path / "config.yaml"
        data = read_config_as_dict(path)
        assert data["Task"] == "mytask"
        assert data["batch_size"] == 16

    def test_ma_dlc_create_cfg_writes_config_yaml(self, tmp_path):
        config = MaDLC_config()
        config.create_cfg(tmp_path, {"Task": "matask"})
        path = tmp_path / "config.yaml"
        assert path.exists()
        data = read_config_as_dict(path)
        assert data["Task"] == "matask"
        assert data["multianimalproject"] is True


class TestModifyTrainTestCfg:
    """Behavior of modify_train_test_cfg (train/test pose config updates)."""

    def test_modify_train_test_cfg_requires_existing_config_path(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        # Non-existent config must not silently succeed
        with pytest.raises(FileNotFoundError):
            modify_train_test_cfg(config_path)

    def test_modify_train_test_cfg_sets_expected_values(self, tmp_path, monkeypatch):
        train_path = tmp_path / "train" / "pytorch_config.yaml"
        test_path = tmp_path / "test" / "pose_cfg.yaml"
        snapshot_folder = tmp_path / "train"
        train_path.parent.mkdir(parents=True)
        test_path.parent.mkdir(parents=True)
        write_config(train_path, {"batch_size": 1, "multi_stage": False, "gradient_masking": False})
        write_config(test_path, {"batch_size": 1, "multi_stage": False, "gradient_masking": False})
        monkeypatch.setattr(
            "deeplabcut.modelzoo.generalized_data_converter.datasets.materialize.compat.return_train_network_path",
            lambda *args, **kwargs: (train_path, test_path, snapshot_folder),
        )
        modify_train_test_cfg(tmp_path / "config.yaml")  # ignored by mock
        for path in (train_path, test_path):
            data = read_config_as_dict(path)
            assert data["multi_stage"] is True
            assert data["batch_size"] == 8
            assert data["gradient_masking"] is True
