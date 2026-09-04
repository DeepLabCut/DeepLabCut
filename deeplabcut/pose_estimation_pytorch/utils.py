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
from __future__ import annotations

import random
import warnings

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

from deeplabcut.pose_estimation_pytorch.config.pose import DetectorConfig, PoseConfig


def fix_seeds(seed: int) -> None:
    """Fixes the random seed for python, numpy and pytorch.

    Args:
        seed: the seed to set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MIN_TORCH_FOR_DETECTOR_MPS = (2, 12)
"""Minimum torch release for running detectors on Apple MPS; older ones hang (DeepLabCut#3155)."""

DETECTOR_MPS_VALIDATED_VARIANTS: frozenset[str] = frozenset({"ssdlite"})
"""Detector variants checked against CPU runs on Apple MPS (ssdlite, torch 2.12.1 and 2.13.0).

Only these resolve to MPS with ``device: auto``; other variants fall back to the CPU. Training the
Faster R-CNN variants on MPS hangs the GPU (fasterrcnn_resnet50_fpn_v2 hard enough to trigger a
macOS watchdog kernel panic) through torchvision's ``roi_align`` backward kernel, fixed in
torchvision 0.29.0 (pytorch/vision#9510) but not validated here yet.
"""


def torch_meets_detector_mps_floor() -> bool:
    """Whether the installed torch is a release >= ``MIN_TORCH_FOR_DETECTOR_MPS``.

    Official pre-releases and unparsable version strings do not qualify. A pre-release
    tag paired with a ``"+git..."`` local segment marks a from-source build of that
    release (PyTorch's own versioning convention, e.g. ``"2.13.0a0+git1234567"``); those
    still qualify, since the source tree already has the release's fixes. Such builds are
    compared by release number alone: PEP 440 orders a pre-release below its final release,
    so a from-source build of the floor version itself (e.g. ``"2.12.0a0+git1234567"``)
    would otherwise be rejected.
    """
    floor = Version(".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS))
    try:
        installed = Version(torch.__version__)
    except InvalidVersion:
        return False
    from_source_build = installed.local is not None and installed.local.startswith("git")
    if installed.is_prerelease and not from_source_build:
        return False
    if from_source_build:
        return installed.release >= floor.release
    return installed >= floor


def is_mps_device(device: str | torch.device | None) -> bool:
    """Whether a device targets Apple MPS (``"mps"``, ``"mps:0"``, ...)."""
    return device is not None and str(device).startswith("mps")


def detector_variant(config: DetectorConfig | dict) -> str | None:
    """Returns the canonical variant name of a detector configuration, or None if unknown."""
    model = config["model"] if isinstance(config, dict) else config.model
    variant = model.get("variant")
    if variant:
        return str(variant)
    if str(model.get("type", "")).lower() == "ssdlite":
        return "ssdlite"
    return None


def detector_mps_supported(variant: str | None) -> bool:
    """Whether MPS is available, torch meets the floor and ``variant`` is validated (None is not)."""
    return (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
        and torch_meets_detector_mps_floor()
        and variant in DETECTOR_MPS_VALIDATED_VARIANTS
    )


def resolve_detector_device(device: str | torch.device | None, variant: str | None) -> str | torch.device | None:
    """Applies the MPS gate to the device a detector is about to be trained on.

    Non-MPS devices and validated variants are returned unchanged; otherwise a warning names the
    reason and the detector falls back to ``"cpu"``.
    """
    if device is None or not is_mps_device(device):
        return device
    if detector_mps_supported(variant):
        return device

    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        reason = f"Detector device {str(device)!r} was requested, but MPS is not available on this machine."
    elif not torch_meets_detector_mps_floor():
        floor = ".".join(str(v) for v in MIN_TORCH_FOR_DETECTOR_MPS)
        reason = (
            f"Running detectors on MPS requires torch >= {floor} (found {torch.__version__}); "
            "older versions are known to hang (DeepLabCut#3155)."
        )
    else:
        reason = (
            f"Detector variant {variant!r} has not been validated on MPS. Training unvalidated detectors on "
            "MPS has been observed to hang the GPU badly enough to trigger a system watchdog reboot "
            "(fasterrcnn_resnet50_fpn_v2 on Apple Silicon; torchvision roi_align backward bug, "
            "pytorch/vision#9510, fixed in torchvision 0.29.0)."
        )
    warnings.warn(
        f"{reason} The detector is trained on the CPU instead. To silence this warning, put the detector on "
        'the CPU explicitly (detector.device: cpu in the model configuration, or device="cpu").',
        UserWarning,
        stacklevel=2,
    )
    return "cpu"


def resolve_device(model_config: PoseConfig | DetectorConfig) -> str:
    """Determines which device should be used from the model config.

    When the device is set to 'auto':
        If an Nvidia GPU is available, selects the device as cuda:0.
        Selects 'mps' if available (on macOS) and the model supports it: resnet pose
        backbones, and detector variants validated on MPS on torch >= 2.12.
        Otherwise, returns 'cpu'.
    Otherwise, simply returns the selected device

    Args:
        model_config (PoseConfig | dict | str | Path): The PyTorch pose configuration.

    Returns:
        the device on which training should be run
    """
    device = model_config.device

    if isinstance(model_config, DetectorConfig):
        supports_mps = detector_mps_supported(detector_variant(model_config))
    else:
        supports_mps = "resnet" in model_config.get("net_type", "")

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif supports_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
