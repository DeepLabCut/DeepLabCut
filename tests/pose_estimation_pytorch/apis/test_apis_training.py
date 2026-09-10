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
"""Tests for the training API."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.apis.training import train, train_network
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.enums import DetectorType
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig, PoseConfig
from deeplabcut.pose_estimation_pytorch.task import Task


def _project_cfg(tmp_path: Path) -> dict:
    return {
        "multianimalproject": False,
        "project_path": str(tmp_path),
        "bodyparts": ["nose"],
        "uniquebodyparts": [],
        "individuals": ["mouse"],
    }


def _minimal_run_config(tmp_path: Path, *, resume_from: str | None = None) -> PoseConfig:
    cfg_path = tmp_path / "pytorch_config.yaml"
    pose_config = make_pytorch_pose_config(_project_cfg(tmp_path), str(cfg_path), net_type="resnet_50")
    if resume_from is not None:
        pose_config.resume_training_from = resume_from
    return pose_config


def _top_down_config(tmp_path: Path) -> PoseConfig:
    """A top-down ResNet pose config; its detector is an SSDLite (a variant validated on MPS)."""
    cfg_path = tmp_path / "pytorch_config.yaml"
    return make_pytorch_pose_config(_project_cfg(tmp_path), str(cfg_path), net_type="resnet_50", top_down=True)


def _make_loader(tmp_path: Path, run_config: PoseConfig | DetectorConfig) -> Mock:
    loader = Mock()
    loader.model_folder = tmp_path
    loader.model_cfg = run_config
    train_dataset = Mock(__len__=Mock(return_value=1))
    valid_dataset = Mock(__len__=Mock(return_value=1))
    loader.create_dataset = Mock(side_effect=[train_dataset, valid_dataset])
    return loader


@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.PoseModel.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_uses_resume_training_from_config(
    mock_build_runner: Mock,
    mock_build_model: Mock,
    mock_build_transforms: Mock,
    tmp_path: Path,
) -> None:
    run_config = _minimal_run_config(tmp_path, resume_from="/train/snapshot-010.pt")
    loader = _make_loader(tmp_path, run_config)

    train(loader=loader, run_config=run_config, task=Task.BOTTOM_UP, device="cpu", snapshot_path=None)

    assert mock_build_runner.call_args.kwargs["snapshot_path"] == "/train/snapshot-010.pt"


# the phrase every gate warning ends with, to tell them apart from unrelated ones
_GATE_WARNING_MARKER = "trained on the CPU instead"
_FRCNN = "fasterrcnn_resnet50_fpn_v2"


def _apple_silicon(monkeypatch, *, mps: bool = True, floor: bool = True) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: mps)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: floor)


def _run_config(kind: str, tmp_path: Path) -> PoseConfig | DetectorConfig:
    if kind == "pose":
        return _top_down_config(tmp_path)
    if kind == "ssdlite":
        return _top_down_config(tmp_path)["detector"]
    detector_config = DetectorConfig.build(
        1, DetectorType.FASTERRCNN_RESNET50_FPN_V2
    )  # variant fasterrcnn_resnet50_fpn_v2
    if kind == "no-variant":
        detector_config["model"]["variant"] = None
    return detector_config


# ---------------------------------------------------------------------------
# train(): the MPS gate for detectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, config_device, requested, env, expected, warns",
    [
        pytest.param("ssdlite", "auto", "mps", {}, "mps", False, id="validated-mps"),
        pytest.param("ssdlite", "auto", None, {}, "mps", False, id="validated-config-auto"),
        pytest.param("ssdlite", "auto", "cpu", {}, "cpu", False, id="explicit-cpu"),
        pytest.param("ssdlite", "auto", "mps", {"floor": False}, "cpu", True, id="torch-below-floor"),
        pytest.param("ssdlite", "auto", "mps", {"mps": False}, "cpu", True, id="mps-not-available"),
        pytest.param("fasterrcnn", "auto", "mps", {}, "cpu", True, id="unvalidated-mps"),
        pytest.param("fasterrcnn", "auto", "mps:0", {}, "cpu", True, id="unvalidated-mps0"),
        pytest.param("fasterrcnn", "auto", torch.device("mps"), {}, "cpu", True, id="unvalidated-torch-device"),
        pytest.param("fasterrcnn", "mps", None, {}, "cpu", True, id="unvalidated-config-mps"),
        pytest.param("fasterrcnn", "auto", None, {}, "cpu", False, id="unvalidated-config-auto"),
        pytest.param("no-variant", "auto", "mps", {}, "cpu", True, id="unknown-variant"),
        pytest.param("pose", "auto", "mps", {}, "mps", False, id="pose-model-untouched"),
    ],
)
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_transforms", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.DETECTORS.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.PoseModel.build", return_value=Mock())
@patch("deeplabcut.pose_estimation_pytorch.apis.training.build_training_runner", return_value=Mock())
def test_train_applies_detector_mps_gate(
    mock_build_runner: Mock,
    mock_build_pose: Mock,
    mock_build_detector: Mock,
    mock_build_transforms: Mock,
    kind: str,
    config_device: str,
    requested,
    env: dict,
    expected: str,
    warns: bool,
    tmp_path: Path,
    monkeypatch,
    recwarn,
) -> None:
    """Detectors reach the training runner on MPS only when validated; the pose model is untouched."""
    _apple_silicon(monkeypatch, **env)
    run_config = _run_config(kind, tmp_path)
    run_config["device"] = config_device
    task = Task.TOP_DOWN if kind == "pose" else Task.DETECT

    train(loader=_make_loader(tmp_path, run_config), run_config=run_config, task=task, device=requested)

    assert mock_build_runner.call_args.kwargs["device"] == expected
    model = mock_build_pose.return_value if kind == "pose" else mock_build_detector.return_value
    model.to.assert_called_with(expected)  # the patched builders are shared across the parametrized cases
    gate_warnings = [str(w.message) for w in recwarn if _GATE_WARNING_MARKER in str(w.message)]
    assert (len(gate_warnings) == 1) is warns
    if kind in ("fasterrcnn", "no-variant") and warns:
        assert "watchdog" in gate_warnings[0]


# ---------------------------------------------------------------------------
# train_network(): detector device inheritance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detector_device, top_level_device, explicit_device, expected_detector_device, expected_detector_arg",
    [
        pytest.param("cpu", "auto", None, "cpu", None, id="explicit-detector-cpu-kept"),
        pytest.param("auto", "cuda:1", None, "cuda:1", None, id="detector-auto-inherits-top-level"),
        pytest.param("auto", "auto", None, "auto", None, id="both-auto"),
        pytest.param("cpu", "auto", "mps", "cpu", "mps", id="explicit-argument-forwarded"),
        pytest.param("cpu", "auto", "auto", "cpu", None, id="auto-argument-keeps-detector-cpu"),
        pytest.param("auto", "cpu", "auto", "auto", None, id="auto-argument-detector-auto-policy"),
    ],
)
@patch("deeplabcut.pose_estimation_pytorch.apis.training.destroy_file_logging")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.setup_file_logging")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.train")
@patch("deeplabcut.pose_estimation_pytorch.apis.training.DLCLoader")
def test_train_network_detector_device_inheritance(
    mock_loader_cls: Mock,
    mock_train: Mock,
    mock_setup_file_logging: Mock,
    mock_destroy_file_logging: Mock,
    detector_device: str,
    top_level_device: str,
    explicit_device: str | None,
    expected_detector_device: str,
    expected_detector_arg: str | None,
    tmp_path: Path,
) -> None:
    """A detector left on auto inherits the top-level device; an explicit detector.device is kept."""
    pose_config = _top_down_config(tmp_path)
    pose_config["device"] = top_level_device
    pose_config["detector"]["device"] = detector_device
    loader = Mock()
    loader.model_cfg = pose_config
    loader.model_folder = tmp_path
    loader.model_config_path = tmp_path / "pytorch_config.yaml"
    mock_loader_cls.return_value = loader

    train_network(config=tmp_path / "config.yaml", device=explicit_device)

    assert mock_train.call_count == 2
    detector_call, pose_call = mock_train.call_args_list
    assert detector_call.kwargs["task"] == Task.DETECT
    assert detector_call.kwargs["run_config"]["device"] == expected_detector_device
    assert detector_call.kwargs["device"] == expected_detector_arg
    assert pose_call.kwargs["task"] == Task.TOP_DOWN
    assert pose_call.kwargs["run_config"]["device"] == top_level_device
    assert pose_call.kwargs["device"] == explicit_device
