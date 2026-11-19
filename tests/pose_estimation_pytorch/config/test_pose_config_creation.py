#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for pose config creation via make_pytorch_pose_config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.config.utils import get_config_folder_path

try:
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig

    build_pose_config = PoseConfig.build
except ImportError:
    from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config as build_pose_config


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _as_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return cfg
    return cfg.to_dict(normalize=True)


SINGLE_ANIMAL_PROJECT = read_config_as_dict(FIXTURES_DIR / "single_animal_project_v0.yaml")
MULTIANIMAL_PROJECT = read_config_as_dict(FIXTURES_DIR / "multianimal_project_v0.yaml")
MULTIANIMAL_TD_PROJECT = copy.deepcopy(MULTIANIMAL_PROJECT)
MULTIANIMAL_TD_PROJECT["uniquebodyparts"] = []


class PoseConfigCase(NamedTuple):
    id: str
    project: dict
    net_type: str
    build_kwargs: dict[str, Any]
    method: str
    bodyparts: list[str]
    individuals: list[str]
    unique_bodyparts: list[str]
    backbone_type: str | None
    detector_type: str | None = None
    head_type: str | None = None


POSE_CONFIG_CASES = [
    PoseConfigCase(
        "single_resnet_50_bu",
        SINGLE_ANIMAL_PROJECT,
        "resnet_50",
        {},
        "bu",
        ["snout", "leftear", "rightear", "tailbase"],
        ["individual_1"],
        [],
        "ResNet",
        head_type="HeatmapHead",
    ),
    PoseConfigCase(
        "single_hrnet_w32_bu",
        SINGLE_ANIMAL_PROJECT,
        "hrnet_w32",
        {},
        "bu",
        ["snout", "leftear", "rightear", "tailbase"],
        ["individual_1"],
        [],
        "HRNet",
        head_type="HeatmapHead",
    ),
    PoseConfigCase(
        "multi_resnet_50_bu",
        MULTIANIMAL_PROJECT,
        "resnet_50",
        {},
        "bu",
        ["nose", "tail"],
        ["mouse1", "mouse2"],
        ["corner1", "corner2"],
        "ResNet",
        head_type="DLCRNetHead",
    ),
    PoseConfigCase(
        "multi_dlcrnet_stride16_ms5_bu",
        MULTIANIMAL_PROJECT,
        "dlcrnet_stride16_ms5",
        {},
        "bu",
        ["nose", "tail"],
        ["mouse1", "mouse2"],
        ["corner1", "corner2"],
        "DLCRNet",
        head_type="DLCRNetHead",
    ),
    PoseConfigCase(
        "multi_dekr_w18_bu",
        MULTIANIMAL_PROJECT,
        "dekr_w18",
        {},
        "bu",
        ["nose", "tail"],
        ["mouse1", "mouse2"],
        ["corner1", "corner2"],
        "HRNet",
        head_type="DEKRHead",
    ),
    PoseConfigCase(
        "multi_resnet_50_td_ssdlite",
        MULTIANIMAL_TD_PROJECT,
        "resnet_50",
        {"top_down": True, "detector_type": "ssdlite"},
        "td",
        ["nose", "tail"],
        ["mouse1", "mouse2"],
        [],
        "ResNet",
        detector_type="SSDLite",
        head_type="HeatmapHead",
    ),
]


@pytest.mark.parametrize("stage", ["train", "inference"])
def test_pose_config_repairs_top_down_crop_not_matching_head(
    stage: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Reproduces configs created by 3.0.1 to 3.0.3 for rtmpose_x (256 crop, 384 head)."""
    cfg = _as_dict(
        build_pose_config(
            SINGLE_ANIMAL_PROJECT,
            tmp_path / "pytorch_config.yaml",
            net_type="rtmpose_x",
            top_down=True,
            save=False,
        )
    )
    cfg["data"][stage]["top_down_crop"].update(width=256, height=256)

    repaired = PoseConfig.from_dict(cfg)

    crop = getattr(repaired.data, stage).top_down_crop
    assert (crop["width"], crop["height"]) == (384, 384)
    assert f"data.{stage}.top_down_crop" in caplog.text


MODEL_YAMLS_WITH_DATA = sorted(
    path
    for path in get_config_folder_path().glob("*/*.yaml")
    if path.parent.name not in ("base", "backbones", "detectors") and read_config_as_dict(path).get("data")
)


def _leaves(cfg: dict, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    leaves = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            leaves.extend(_leaves(value, (*prefix, key)))
        else:
            leaves.append(((*prefix, key), value))
    return leaves


def _as_lists(value: Any) -> Any:
    """Typed configs may store YAML lists as tuples; compare them as lists."""
    if isinstance(value, (list, tuple)):
        return [_as_lists(v) for v in value]
    return value


@pytest.mark.parametrize("model_yaml", MODEL_YAMLS_WITH_DATA, ids=lambda p: p.stem)
def test_model_yaml_data_takes_precedence_over_augmentation_defaults(model_yaml: Path, tmp_path: Path) -> None:
    """Values in a model YAML's `data` section must not be overwritten by the defaults."""
    model_cfg = read_config_as_dict(model_yaml)
    ctd = str(model_cfg.get("method", "bu")).lower() == "ctd"
    cfg = _as_dict(
        build_pose_config(
            MULTIANIMAL_TD_PROJECT,
            tmp_path / "pytorch_config.yaml",
            net_type=model_yaml.stem,
            top_down=False,
            save=False,
            ctd_conditions=1 if ctd else None,
        )
    )

    for path, expected in _leaves(model_cfg["data"]):
        value = cfg["data"]
        for key in path:
            value = value[key]
        assert _as_lists(value) == _as_lists(expected), f"data.{'.'.join(path)}"


@pytest.mark.parametrize("case", POSE_CONFIG_CASES, ids=lambda c: c.id)
def test_pose_config_creation(case: PoseConfigCase, tmp_path: Path) -> None:
    build_kwargs = dict(case.build_kwargs)
    cfg = _as_dict(
        build_pose_config(
            case.project,
            tmp_path / "pytorch_config.yaml",
            net_type=case.net_type,
            save=False,
            top_down=build_kwargs.pop("top_down", False),
            **build_kwargs,
        )
    )
    meta = cfg["metadata"]

    assert str(cfg["net_type"]) == case.net_type
    assert str(cfg["method"]).lower() == case.method
    assert meta["bodyparts"] == case.bodyparts
    if case.project["multianimalproject"]:
        assert meta["individuals"] == case.individuals
    else:
        assert len(meta["individuals"]) == 1
    assert (meta.get("unique_bodyparts") or []) == case.unique_bodyparts

    if case.backbone_type is not None:
        assert cfg["model"]["backbone"]["type"] == case.backbone_type

    if case.head_type is not None:
        assert cfg["model"]["heads"]["bodypart"]["type"] == case.head_type

    if case.detector_type is not None:
        detector = cfg["detector"]
        assert detector is not None
        assert detector["model"]["type"] == case.detector_type
        assert detector["device"] == "auto"
        assert detector["model"]["box_score_thresh"] == 0.01
        assert cfg["data"]["train"].get("top_down_crop") is not None
    else:
        assert cfg.get("detector") is None


def test_pose_config_save_writes_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "pytorch_config.yaml"
    cfg = build_pose_config(
        SINGLE_ANIMAL_PROJECT,
        config_path,
        net_type="resnet_50",
        top_down=False,
        save=True,
    )
    assert config_path.is_file()
    saved = read_config_as_dict(config_path)
    cfg_dict = _as_dict(cfg)
    assert saved["net_type"] == cfg_dict["net_type"]
    assert saved["metadata"]["bodyparts"] == cfg_dict["metadata"]["bodyparts"]


@pytest.mark.parametrize("project", [SINGLE_ANIMAL_PROJECT, MULTIANIMAL_PROJECT], ids=["single", "multi"])
def test_default_net_type_used_when_net_type_is_none(project: dict) -> None:
    cfg = build_pose_config(project, "pytorch_config.yaml", net_type=None, top_down=False)
    assert str(_as_dict(cfg)["net_type"]) == "resnet_50"


def test_weight_init_is_written_to_train_settings() -> None:
    project_config = {
        "project_path": "/test/project",
        "multianimalproject": False,
        "identity": False,
        "bodyparts": ["nose", "ear"],
    }
    snapshot_path = Path("/tmp/snapshot-010.pt")
    weight_init = WeightInitialization(snapshot_path=snapshot_path)

    cfg = build_pose_config(
        project_config,
        "pytorch_config.yaml",
        net_type="resnet_50",
        weight_init=weight_init,
        top_down=False,
    )

    saved = _as_dict(cfg)["train_settings"]["weight_init"]
    assert saved["snapshot_path"] == str(snapshot_path)
