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
"""Tests for the MPS hardware gate for object detectors (pose_estimation_pytorch.utils).

Every test fakes the hardware, so the suite passes on machines without MPS or CUDA.
"""

from __future__ import annotations

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.utils as dlc_utils
from deeplabcut.pose_estimation_pytorch.config.model import DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig
from deeplabcut.pose_estimation_pytorch.utils import (
    detector_mps_supported,
    detector_variant,
    is_mps_device,
    resolve_detector_device,
    resolve_device,
)

FRCNN = "fasterrcnn_resnet50_fpn_v2"


class FakePoseConfig:
    """Duck-typed pose config: .device and .get("net_type")."""

    def __init__(self, device="auto", net_type="resnet_50"):
        self.device = device
        self.net_type = net_type

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_detector(device: str = "auto", type_: str = "SSDLite", variant: str | None = None) -> DetectorConfig:
    return DetectorConfig(model=DetectorModelConfig(type=type_, variant=variant), device=device)


@pytest.fixture
def apple_silicon(monkeypatch):
    """No CUDA, MPS built and available, torch above the floor, ssdlite the only validated variant."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: True)
    monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset({"ssdlite"}))


def _break(monkeypatch, condition: str) -> None:
    if condition == "registry":
        monkeypatch.setattr(dlc_utils, "DETECTOR_MPS_VALIDATED_VARIANTS", frozenset())
    elif condition == "floor":
        monkeypatch.setattr(dlc_utils, "torch_meets_detector_mps_floor", lambda: False)
    elif condition == "mps":
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    elif condition == "mps-built":
        monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)


def test_shipped_registry():
    assert dlc_utils.DETECTOR_MPS_VALIDATED_VARIANTS == {"ssdlite"}


def test_resolve_device_pose_models_unchanged(apple_silicon):
    assert resolve_device(FakePoseConfig(net_type="resnet_50")) == "mps"
    assert resolve_device(FakePoseConfig(net_type="hrnet_w32")) == "cpu"
    assert resolve_device(FakePoseConfig(device="cuda:1")) == "cuda:1"


def test_resolve_device_detector_auto(apple_silicon, monkeypatch):
    assert resolve_device(make_detector()) == "mps"
    assert resolve_device(make_detector(type_="FasterRCNN", variant=FRCNN)) == "cpu"
    assert resolve_device(make_detector(device="cpu")) == "cpu"  # explicit devices are returned verbatim
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(make_detector()) == "cuda"


@pytest.mark.parametrize("condition", ["registry", "floor", "mps", "mps-built"])
def test_detector_needs_every_condition(apple_silicon, monkeypatch, condition):
    assert detector_mps_supported("ssdlite") is True
    _break(monkeypatch, condition)
    assert detector_mps_supported("ssdlite") is False
    assert resolve_device(make_detector()) == "cpu"


def test_detector_variant():
    assert detector_variant(make_detector(type_="FasterRCNN", variant=FRCNN)) == FRCNN
    assert detector_variant(make_detector()) == "ssdlite"
    assert detector_variant(make_detector(type_="FasterRCNN")) is None  # unknown -> unvalidated
    assert detector_variant({"model": {"type": "SSDLite", "variant": None}}) == "ssdlite"


@pytest.mark.parametrize(
    "device, expected",
    [("mps", True), ("mps:0", True), (torch.device("mps"), True), ("cpu", False), ("cuda:0", False), (None, False)],
)
def test_is_mps_device(device, expected):
    assert is_mps_device(device) is expected


@pytest.mark.parametrize(
    "version, expected",
    [
        ("2.12.0", True),
        ("2.13.0+cu128", True),
        ("2.11.9", False),
        ("2.13.0a0+git1234", True),  # from-source build of the 2.13.0 release
        ("2.11.9a0+git1234", False),  # from-source build, but below the floor
        ("2.13.0rc1", False),  # official pre-release, not a from-source build
        ("2.13.0.dev20260101+cu124", False),  # nightly wheel, not a from-source build
        ("not-a-version", False),
    ],
)
def test_torch_meets_detector_mps_floor(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)
    assert dlc_utils.torch_meets_detector_mps_floor() is expected


def test_resolve_detector_device_keeps_allowed_devices(apple_silicon, recwarn):
    assert resolve_detector_device("cpu", FRCNN) == "cpu"
    assert resolve_detector_device(None, None) is None
    assert resolve_detector_device("mps", "ssdlite") == "mps"
    assert resolve_detector_device("mps:0", "ssdlite") == "mps:0"
    assert recwarn.list == []


@pytest.mark.parametrize(
    "condition, variant, expected_text",
    [
        ("registry", FRCNN, [FRCNN, "not been validated", "watchdog"]),
        ("registry", None, ["None", "not been validated"]),
        ("floor", "ssdlite", ["2.12"]),
        ("mps", "ssdlite", ["not available"]),
    ],
)
def test_resolve_detector_device_falls_back_to_cpu_with_reason(
    apple_silicon, monkeypatch, condition, variant, expected_text
):
    _break(monkeypatch, condition)
    with pytest.warns(UserWarning) as record:
        assert resolve_detector_device("mps", variant) == "cpu"
    (message,) = [str(w.message) for w in record]
    for text in expected_text:
        assert text in message
    assert "trained on the CPU instead" in message
