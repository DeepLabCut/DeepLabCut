from __future__ import annotations

import functools
from enum import Enum

_LEGACY_NET_TYPE_ALIASES = {
    "dlcrnet_ms5": "dlcrnet_stride16_ms5",
}
_TOP_DOWN_PREFIX = "top_down_"


class MethodType(str, Enum):
    """Enumeration of pose estimation method types."""

    BOTTOM_UP = "bu"
    TOP_DOWN = "td"
    CONDITIONAL_TOP_DOWN = "ctd"


class NetType(str, Enum):
    """Enumeration of network architecture types as stored in configs.

    Note:
        Aliases (e.g. ``top_down_resnet_50``) are user-facing names that map to
        a canonical member plus an optional top-down flag. See ``alias``,
        ``from_alias``, and ``available_aliases``
    """

    RESNET_50 = "resnet_50"
    RESNET_101 = "resnet_101"

    HRNET_W18 = "hrnet_w18"
    HRNET_W32 = "hrnet_w32"
    HRNET_W48 = "hrnet_w48"

    CSPNEXT_S = "cspnext_s"
    CSPNEXT_M = "cspnext_m"
    CSPNEXT_X = "cspnext_x"

    DEKR_W18 = "dekr_w18"
    DEKR_W32 = "dekr_w32"
    DEKR_W48 = "dekr_w48"

    CTD_COAM_W32 = "ctd_coam_w32"
    CTD_COAM_W48 = "ctd_coam_w48"
    CTD_COAM_W48_HUMAN = "ctd_coam_w48_human"
    CTD_PRENET_HRNET_W32 = "ctd_prenet_hrnet_w32"
    CTD_PRENET_HRNET_W48 = "ctd_prenet_hrnet_w48"
    CTD_PRENET_RTMPOSE_S = "ctd_prenet_rtmpose_s"
    CTD_PRENET_RTMPOSE_M = "ctd_prenet_rtmpose_m"
    CTD_PRENET_RTMPOSE_X = "ctd_prenet_rtmpose_x"
    CTD_PRENET_RTMPOSE_X_HUMAN = "ctd_prenet_rtmpose_x_human"

    DLCRNET_STRIDE16_MS5 = "dlcrnet_stride16_ms5"
    DLCRNET_STRIDE32_MS5 = "dlcrnet_stride32_ms5"

    RTMPOSE_S = "rtmpose_s"
    RTMPOSE_M = "rtmpose_m"
    RTMPOSE_X = "rtmpose_x"

    ANIMALTOKENPOSE_BASE = "animaltokenpose_base"

    @functools.cached_property
    def is_backbone(self) -> bool:
        from deeplabcut.pose_estimation_pytorch.config.utils import (
            get_config_folder_path,
            load_backbones,
        )

        return self.value in frozenset(load_backbones(get_config_folder_path()))

    def alias(self, *, top_down: bool = False) -> str:
        """User-facing name (e.g. ``top_down_resnet_50`` for backbone + TD)."""
        if top_down and self.is_backbone:
            return f"{_TOP_DOWN_PREFIX}{self.value}"
        return self.value

    @classmethod
    def from_alias(cls, label: str) -> tuple[NetType, bool]:
        """Parse user-facing / legacy name → (canonical enum, top_down)."""
        label = _LEGACY_NET_TYPE_ALIASES.get(label, label)
        label_has_td_prefix = False
        if label.startswith(_TOP_DOWN_PREFIX):
            label = label.removeprefix(_TOP_DOWN_PREFIX)
            label_has_td_prefix = True
        return cls(label), label_has_td_prefix

    @classmethod
    def available_aliases(cls) -> list[str]:
        """All selectable model names for GUI / docs / ``create_training_dataset``."""
        labels: list[str] = []
        for net_type in cls:
            labels.append(net_type.alias(top_down=False))
            if net_type.is_backbone:
                labels.append(net_type.alias(top_down=True))
        return sorted(labels)


class DetectorType(str, Enum):
    """Enumeration of detector types."""

    SSDLITE = "ssdlite"
    FASTERRCNN_RESNET50_FPN_V2 = "fasterrcnn_resnet50_fpn_v2"
    FASTERRCNN_MOBILENET_V3_LARGE_FPN = "fasterrcnn_mobilenet_v3_large_fpn"


class DatasetType(str, Enum):
    """Enumeration of dataset types."""

    # TODO @deruyter92 2026-02-05: Add other dataset types as needed.
    MULTIANIMAL_IMGAUG = "multi-animal-imgaug"
