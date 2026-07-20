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

import pytest

from deeplabcut.pose_estimation_pytorch import data as data_module
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


def _ctd_model_conditions_config() -> ConditionsModelConfig:
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


class _FakeDLCLoader(data_module.DLCLoader):
    """Minimal DLCLoader stand-in so isinstance(..., DLCLoader) succeeds."""

    def __init__(self, conditions):
        self.pose_task = Task.COND_TOP_DOWN
        self.model_cfg = {"inference": {"conditions": conditions}}
        self.image_root = Path("/images")
        self.project_root = Path("/project")


# --- ConditionsConfig.build ---------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_type, check",
    [
        pytest.param(None, type(None), lambda c: c is None, id="none"),
        pytest.param(
            "/path/to/preds.h5",
            ConditionsFileConfig,
            lambda c: c.filepath == Path("/path/to/preds.h5"),
            id="path-str",
        ),
        pytest.param(
            Path("/path/to/preds.json"),
            ConditionsFileConfig,
            lambda c: c.filepath == Path("/path/to/preds.json"),
            id="path-obj",
        ),
        pytest.param(
            {"filepath": "/preds.h5"},
            ConditionsFileConfig,
            lambda c: c.filepath == Path("/preds.h5"),
            id="filepath-dict",
        ),
        pytest.param(
            {"shuffle": 3},
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 3,
            id="shuffle-dict",
        ),
        pytest.param(
            {"shuffle": 1, "snapshot_index": -1},
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 1 and c.snapshot_index == -1,
            id="shuffle-snapshot-index",
        ),
        pytest.param(
            {
                "config_path": "/bu/pytorch_config.yaml",
                "snapshot_path": "/bu/snapshot.pt",
            },
            ConditionsModelConfig,
            lambda c: c.config_path == Path("/bu/pytorch_config.yaml"),
            id="model-dict",
        ),
        pytest.param(
            {"source": "shuffle", "shuffle": 9},
            ConditionsShuffleConfig,
            lambda c: c.shuffle == 9,
            id="explicit-source-shuffle",
        ),
    ],
)
def test_conditions_config_build(raw, expected_type, check):
    built = ConditionsConfig.build(raw)
    assert isinstance(built, expected_type)
    assert check(built)


def test_conditions_config_build_passthrough():
    ctd_conditions = _ctd_model_conditions_config()
    assert ConditionsConfig.build(ctd_conditions) is ctd_conditions


def test_conditions_config_build_rejects_ambiguous_dict():
    with pytest.raises(ValueError, match="Cannot determine conditions source"):
        ConditionsConfig.build({"unexpected": 1})


def test_conditions_config_build_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Cannot build"):
        ConditionsConfig.build(42)  # type: ignore[arg-type]


# --- resolve_from_conditions --------------------------------------------------


def test_resolve_from_conditions_model_identity():
    ctd_conditions = _ctd_model_conditions_config()
    assert ConditionsModelConfig.resolve_from_conditions(ctd_conditions) is ctd_conditions


def test_resolve_from_conditions_rejects_file_config():
    with pytest.raises(ValueError, match="evaluation only"):
        ConditionsModelConfig.resolve_from_conditions(ConditionsFileConfig(filepath=Path("/preds.h5")))


def test_resolve_from_conditions_rejects_filepath_dict():
    with pytest.raises(ValueError, match="evaluation only"):
        ConditionsModelConfig.resolve_from_conditions({"filepath": "/preds.h5"})


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param({"shuffle": 1}, id="dict"),
        pytest.param(ConditionsShuffleConfig(shuffle=1), id="typed"),
    ],
)
def test_resolve_from_conditions_shuffle_requires_config(conditions):
    with pytest.raises(ValueError, match="no project config"):
        ConditionsModelConfig.resolve_from_conditions(conditions)


@pytest.mark.parametrize(
    "conditions",
    [
        pytest.param(
            ConditionsShuffleConfig(shuffle=7, snapshot_index=-1),
            id="typed",
        ),
        pytest.param(
            {"shuffle": 7, "snapshot_index": -1},
            id="dict",
        ),
    ],
)
def test_resolve_from_conditions_shuffle_with_config(conditions, monkeypatch):
    """Shuffle forms are forwarded to from_shuffle with the project config injected."""
    mock_from_shuffle = Mock(return_value=_ctd_model_conditions_config())
    monkeypatch.setattr(
        ConditionsModelConfig,
        "from_shuffle",
        classmethod(lambda cls, **kwargs: mock_from_shuffle(**kwargs)),
    )

    ConditionsModelConfig.resolve_from_conditions(
        conditions,
        config="/project/config.yaml",
    )

    mock_from_shuffle.assert_called_once_with(
        config=Path("/project/config.yaml"),
        shuffle=7,
        trainset_index=0,
        modelprefix="",
        snapshot=None,
        snapshot_index=-1,
    )


def test_resolve_from_conditions_uses_embedded_shuffle_config(monkeypatch):
    mock_from_shuffle = Mock(return_value=_ctd_model_conditions_config())
    monkeypatch.setattr(
        ConditionsModelConfig,
        "from_shuffle",
        classmethod(lambda cls, **kwargs: mock_from_shuffle(**kwargs)),
    )

    ConditionsModelConfig.resolve_from_conditions(
        ConditionsShuffleConfig(shuffle=3, config=Path("/embedded/config.yaml")),
    )

    mock_from_shuffle.assert_called_once_with(
        config=Path("/embedded/config.yaml"),
        shuffle=3,
        trainset_index=0,
        modelprefix="",
        snapshot=None,
        snapshot_index=None,
    )


# --- InferenceConfig + _add_ctd_conditions / PoseConfig ------------------------


def test_inference_config_accepts_path_string():
    """Regression: bare path strings must validate (PoseConfig / YAML file form)."""
    cfg = InferenceConfig(conditions="/path/to/bu_predictions.h5")
    assert isinstance(cfg.conditions, ConditionsFileConfig)
    assert cfg.conditions.filepath == Path("/path/to/bu_predictions.h5")


@pytest.mark.parametrize(
    "ctd_conditions, expected",
    [
        pytest.param(5, {"shuffle": 5}, id="int"),
        pytest.param((1, -1), {"shuffle": 1, "snapshot_index": -1}, id="tuple-index"),
        pytest.param(
            (2, "snapshot-best-150.pt"),
            {"shuffle": 2, "snapshot": "snapshot-best-150.pt"},
            id="tuple-name",
        ),
    ],
)
def test_add_ctd_conditions_shuffle_forms(ctd_conditions, expected):
    model_cfg: dict = {"inference": {}}
    _add_ctd_conditions(model_cfg, ctd_conditions)
    assert model_cfg["inference"]["conditions"] == expected
    built = ConditionsConfig.build(model_cfg["inference"]["conditions"])
    assert isinstance(built, ConditionsShuffleConfig)


def test_add_ctd_conditions_file_path(tmp_path: Path):
    preds = tmp_path / "bu_predictions.h5"
    preds.write_bytes(b"")

    model_cfg: dict = {"inference": {}}
    _add_ctd_conditions(model_cfg, preds)
    assert model_cfg["inference"]["conditions"] == str(preds.resolve())

    built = ConditionsConfig.build(model_cfg["inference"]["conditions"])
    assert isinstance(built, ConditionsFileConfig)
    assert built.filepath == preds.resolve()


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
    loader = _ctd_loader(conditions="/preds.h5")  # YAML-like raw path string

    with patch("deeplabcut.pose_estimation_pytorch.apis.ctd.CondFromFile") as mock_cond:
        load_conditions_for_evaluation(loader, ["img.png"])

    mock_cond.assert_called_once_with(filepath=Path("/preds.h5"))
    mock_cond.return_value.load_conditions.assert_called_once_with(["img.png"], path_prefix=loader.image_root)


def test_load_conditions_for_evaluation_from_shuffle():
    loader = _ctd_loader(
        conditions={
            "shuffle": 4,
            "config": "/project/config.yaml",
            "snapshot": "snapshot-100.pt",
        }
    )

    with patch("deeplabcut.pose_estimation_pytorch.apis.ctd.CondFromFile") as mock_cond:
        load_conditions_for_evaluation(loader, ["img.png"])

    mock_cond.assert_called_once_with(
        config=Path("/project/config.yaml"),
        shuffle=4,
        trainset_index=0,
        modelprefix="",
        snapshot="snapshot-100.pt",
        snapshot_index=None,
    )


def test_load_conditions_for_evaluation_injects_dlcloader_project_config():
    loader = _FakeDLCLoader(conditions={"shuffle": 4})

    with patch("deeplabcut.pose_estimation_pytorch.apis.ctd.CondFromFile") as mock_cond:
        load_conditions_for_evaluation(loader, ["img.png"])

    mock_cond.assert_called_once_with(
        config=Path("/project/config.yaml"),
        shuffle=4,
        trainset_index=0,
        modelprefix="",
        snapshot=None,
        snapshot_index=None,
    )


def test_load_conditions_for_evaluation_rejects_model():
    with pytest.raises(ValueError, match="Evaluation accepts file paths or shuffle refs"):
        load_conditions_for_evaluation(
            _ctd_loader(conditions=_ctd_model_conditions_config()),
            ["img.png"],
        )


def test_load_conditions_for_evaluation_rejects_none():
    with pytest.raises(ValueError, match="Got None"):
        load_conditions_for_evaluation(_ctd_loader(conditions=None), ["img.png"])


def test_load_conditions_for_evaluation_rejects_non_ctd():
    loader = _ctd_loader(conditions={"shuffle": 1})
    loader.pose_task = Task.BOTTOM_UP
    with pytest.raises(ValueError, match="only be loaded for CTD"):
        load_conditions_for_evaluation(loader, ["img.png"])
