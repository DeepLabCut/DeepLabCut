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
"""Tests for typed CTD conditions config and evaluation loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.apis.ctd import load_conditions_for_evaluation
from deeplabcut.pose_estimation_pytorch.config.ctd_conditions import (
    ConditionsConfig,
    ConditionsFileConfig,
    ConditionsModelConfig,
    ConditionsShuffleConfig,
)
from deeplabcut.pose_estimation_pytorch.config.inference import InferenceConfig
from deeplabcut.pose_estimation_pytorch.config.make_pose_config import _add_ctd_conditions
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
from deeplabcut.pose_estimation_pytorch.task import Task


def _model_cfg() -> ConditionsModelConfig:
    return ConditionsModelConfig(
        config_path=Path("/bu/pytorch_config.yaml"),
        snapshot_path=Path("/bu/snapshot-best.pt"),
        scorer="DLC_resnet50",
    )


def _ctd_project(tmp_path: Path) -> dict:
    return {
        "multianimalproject": True,
        "project_path": str(tmp_path),
        "bodyparts": "MULTI!",
        "multianimalbodyparts": ["nose", "tail"],
        "uniquebodyparts": [],
        "individuals": ["mouse1", "mouse2"],
        "identity": False,
    }


def _ctd_loader(*, conditions) -> Mock:
    loader = Mock()
    loader.pose_task = Task.COND_TOP_DOWN
    loader.model_cfg = {"inference": {"conditions": conditions}}
    loader.image_root = Path("/images")
    return loader


# --- ConditionsConfig.build ---------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_type, check",
    [
        (None, type(None), lambda c: c is None),
        ("/path/to/preds.h5", ConditionsFileConfig, lambda c: c.filepath == Path("/path/to/preds.h5")),
        (Path("/path/to/preds.json"), ConditionsFileConfig, lambda c: c.filepath == Path("/path/to/preds.json")),
        ({"filepath": "/preds.h5"}, ConditionsFileConfig, lambda c: c.filepath == Path("/preds.h5")),
        ({"shuffle": 3}, ConditionsShuffleConfig, lambda c: c.shuffle == 3),
        (
            {"shuffle": 1, "snapshot_index": -1},
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 1 and c.snapshot_index == -1,
        ),
        (
            {
                "config_path": "/bu/pytorch_config.yaml",
                "snapshot_path": "/bu/snapshot.pt",
            },
            ConditionsModelConfig,
            lambda c: c.config_path == Path("/bu/pytorch_config.yaml"),
        ),
    ],
)
def test_conditions_config_build(raw, expected_type, check):
    built = ConditionsConfig.build(raw)
    assert isinstance(built, expected_type)
    assert check(built)


def test_conditions_config_build_passthrough():
    model = _model_cfg()
    assert ConditionsConfig.build(model) is model


def test_conditions_config_build_rejects_ambiguous_dict():
    with pytest.raises(ValueError, match="Cannot determine conditions source"):
        ConditionsConfig.build({"unexpected": 1})


def test_conditions_config_build_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Cannot build"):
        ConditionsConfig.build(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "conditions, expected",
    [
        (_model_cfg(), True),
        (ConditionsShuffleConfig(shuffle=1), False),
        (ConditionsFileConfig(filepath=Path("/x.h5")), False),
    ],
)
def test_affords_bu_inference(conditions, expected):
    assert conditions.affords_bu_inference is expected


# --- resolve_from_conditions --------------------------------------------------


def test_resolve_from_conditions_model_identity():
    model = _model_cfg()
    assert ConditionsModelConfig.resolve_from_conditions(model) is model


def test_resolve_from_conditions_rejects_file_config():
    with pytest.raises(ValueError, match="evaluation only"):
        ConditionsModelConfig.resolve_from_conditions(ConditionsFileConfig(filepath=Path("/preds.h5")))


def test_resolve_from_conditions_rejects_filepath_dict():
    with pytest.raises(ValueError, match="evaluation only"):
        ConditionsModelConfig.resolve_from_conditions({"filepath": "/preds.h5"})


@pytest.mark.parametrize(
    "conditions",
    [
        {"shuffle": 1},
        ConditionsShuffleConfig(shuffle=1),
    ],
)
def test_resolve_from_conditions_shuffle_requires_config(conditions):
    with pytest.raises(ValueError, match="no project config"):
        ConditionsModelConfig.resolve_from_conditions(conditions)


@pytest.mark.parametrize(
    "conditions",
    [
        ConditionsShuffleConfig(shuffle=7, snapshot_index=-1),
        {"shuffle": 7, "snapshot_index": -1},
    ],
)
def test_resolve_from_conditions_shuffle_with_config(conditions, monkeypatch):
    resolved = _model_cfg()
    calls: list[dict] = []

    @classmethod
    def fake_from_shuffle(cls, **kwargs):
        calls.append(kwargs)
        return resolved

    monkeypatch.setattr(ConditionsModelConfig, "from_shuffle", fake_from_shuffle)

    out = ConditionsModelConfig.resolve_from_conditions(
        conditions,
        config="/project/config.yaml",
    )
    assert out is resolved
    assert calls == [
        {
            "config": Path("/project/config.yaml"),
            "shuffle": 7,
            "trainset_index": 0,
            "modelprefix": "",
            "snapshot": None,
            "snapshot_index": -1,
        }
    ]


# --- InferenceConfig + _add_ctd_conditions / PoseConfig ------------------------


@pytest.mark.parametrize(
    "raw, expected_type, check",
    [
        ("/path/to/bu_predictions.h5", ConditionsFileConfig, lambda c: True),
        (
            {"shuffle": 2, "snapshot": "snapshot-10.pt"},
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 2,
        ),
    ],
)
def test_inference_config_normalizes_conditions(raw, expected_type, check):
    cfg = InferenceConfig(conditions=raw)
    assert isinstance(cfg.conditions, expected_type)
    assert check(cfg.conditions)


@pytest.mark.parametrize(
    "ctd_conditions, expected_type, check",
    [
        (5, ConditionsShuffleConfig, lambda c: c.shuffle == 5),
        ((1, -1), ConditionsShuffleConfig, lambda c: c.shuffle == 1 and c.snapshot_index == -1),
        (
            (2, "snapshot-best-150.pt"),
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 2 and c.snapshot == "snapshot-best-150.pt",
        ),
    ],
)
def test_add_ctd_conditions_shuffle_forms(ctd_conditions, expected_type, check):
    model_cfg: dict = {"inference": {}}
    _add_ctd_conditions(model_cfg, ctd_conditions)
    built = ConditionsConfig.build(model_cfg["inference"]["conditions"])
    assert isinstance(built, expected_type)
    assert check(built)
    assert isinstance(
        InferenceConfig(conditions=model_cfg["inference"]["conditions"]).conditions,
        expected_type,
    )


def test_add_ctd_conditions_file_path(tmp_path: Path):
    preds = tmp_path / "bu_predictions.h5"
    preds.write_bytes(b"")

    model_cfg: dict = {"inference": {}}
    _add_ctd_conditions(model_cfg, preds)
    assert model_cfg["inference"]["conditions"] == str(preds.resolve())

    built = ConditionsConfig.build(model_cfg["inference"]["conditions"])
    assert isinstance(built, ConditionsFileConfig)
    assert built.filepath == preds.resolve()

    typed = InferenceConfig(conditions=model_cfg["inference"]["conditions"])
    assert isinstance(typed.conditions, ConditionsFileConfig)


def test_pose_config_build_ctd_with_shuffle(tmp_path: Path):
    pose_cfg = PoseConfig.build(
        _ctd_project(tmp_path),
        tmp_path / "pytorch_config.yaml",
        top_down=False,
        net_type="ctd_coam_w32",
        ctd_conditions=(1, -1),
    )
    assert isinstance(pose_cfg.inference.conditions, ConditionsShuffleConfig)
    assert pose_cfg.inference.conditions.shuffle == 1
    assert pose_cfg.inference.conditions.snapshot_index == -1


def test_pose_config_build_ctd_with_file(tmp_path: Path):
    preds = tmp_path / "conditions.h5"
    preds.write_bytes(b"")
    pose_cfg = PoseConfig.build(
        _ctd_project(tmp_path),
        tmp_path / "pytorch_config.yaml",
        top_down=False,
        net_type="ctd_coam_w32",
        ctd_conditions=preds,
    )
    assert isinstance(pose_cfg.inference.conditions, ConditionsFileConfig)
    assert pose_cfg.inference.conditions.filepath == preds.resolve()


# --- load_conditions_for_evaluation -------------------------------------------


def test_load_conditions_for_evaluation_from_file():
    loader = _ctd_loader(conditions=ConditionsFileConfig(filepath=Path("/preds.h5")))
    expected = {"img.png": np.zeros((1, 2, 3))}

    with patch("deeplabcut.pose_estimation_pytorch.apis.ctd.CondFromFile") as mock_cond:
        mock_cond.return_value.load_conditions.return_value = expected
        out = load_conditions_for_evaluation(loader, ["img.png"])

    mock_cond.assert_called_once_with(filepath=Path("/preds.h5"))
    mock_cond.return_value.load_conditions.assert_called_once_with(["img.png"], path_prefix=loader.image_root)
    assert out is expected


def test_load_conditions_for_evaluation_from_shuffle():
    conditions = ConditionsShuffleConfig(
        shuffle=4,
        config=Path("/project/config.yaml"),
        snapshot="snapshot-100.pt",
    )
    loader = _ctd_loader(conditions=conditions)
    expected = {"img.png": np.ones((1, 2, 3))}

    with patch("deeplabcut.pose_estimation_pytorch.apis.ctd.CondFromFile") as mock_cond:
        mock_cond.return_value.load_conditions.return_value = expected
        out = load_conditions_for_evaluation(loader, ["img.png"])

    mock_cond.assert_called_once_with(
        config=Path("/project/config.yaml"),
        shuffle=4,
        trainset_index=0,
        modelprefix="",
        snapshot="snapshot-100.pt",
        snapshot_index=None,
    )
    assert out is expected


def test_load_conditions_for_evaluation_rejects_model():
    with pytest.raises(ValueError, match="Misconfigured conditions|Model config|Evaluation accepts"):
        load_conditions_for_evaluation(_ctd_loader(conditions=_model_cfg()), ["img.png"])


def test_load_conditions_for_evaluation_rejects_none():
    with pytest.raises(ValueError, match="Got None|requires conditions"):
        load_conditions_for_evaluation(_ctd_loader(conditions=None), ["img.png"])


def test_load_conditions_for_evaluation_rejects_non_ctd():
    loader = _ctd_loader(conditions={"shuffle": 1})
    loader.pose_task = Task.BOTTOM_UP
    with pytest.raises(ValueError, match="only be loaded for CTD"):
        load_conditions_for_evaluation(loader, ["img.png"])
