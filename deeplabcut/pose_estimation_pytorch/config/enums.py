from enum import Enum


class MethodType(str, Enum):
    """Enumeration of pose estimation method types."""

    BOTTOM_UP = "bu"
    TOP_DOWN = "td"
    CONDITIONAL_TOP_DOWN = "ctd"


class NetType(str, Enum):
    """Enumeration of network architecture types."""

    # ResNet variants (bottom-up)
    RESNET_50 = "resnet_50"
    RESNET_101 = "resnet_101"

    # ResNet variants (top-down)
    TOP_DOWN_RESNET_50 = "top_down_resnet_50"
    TOP_DOWN_RESNET_101 = "top_down_resnet_101"

    # HRNet variants (bottom-up)
    HRNET_W18 = "hrnet_w18"
    HRNET_W32 = "hrnet_w32"
    HRNET_W48 = "hrnet_w48"

    # HRNet variants (top-down)
    TOP_DOWN_HRNET_W18 = "top_down_hrnet_w18"
    TOP_DOWN_HRNET_W32 = "top_down_hrnet_w32"
    TOP_DOWN_HRNET_W48 = "top_down_hrnet_w48"

    # CSPNeXt variants (bottom-up)
    CSPNEXT_S = "cspnext_s"
    CSPNEXT_M = "cspnext_m"
    CSPNEXT_X = "cspnext_x"

    # CSPNeXt variants (top-down)
    TOP_DOWN_CSPNEXT_S = "top_down_cspnext_s"
    TOP_DOWN_CSPNEXT_M = "top_down_cspnext_m"
    TOP_DOWN_CSPNEXT_X = "top_down_cspnext_x"

    # DEKR variants (bottom-up with HRNet backbone)
    DEKR_W18 = "dekr_w18"
    DEKR_W32 = "dekr_w32"
    DEKR_W48 = "dekr_w48"

    # BUCTD variants (Conditional Top-Down)
    CTD_COAM_W32 = "ctd_coam_w32"
    CTD_COAM_W48 = "ctd_coam_w48"
    CTD_COAM_W48_HUMAN = "ctd_coam_w48_human"
    CTD_PRENET_HRNET_W32 = "ctd_prenet_hrnet_w32"
    CTD_PRENET_HRNET_W48 = "ctd_prenet_hrnet_w48"
    CTD_PRENET_RTMPOSE_S = "ctd_prenet_rtmpose_s"
    CTD_PRENET_RTMPOSE_M = "ctd_prenet_rtmpose_m"
    CTD_PRENET_RTMPOSE_X = "ctd_prenet_rtmpose_x"
    CTD_PRENET_RTMPOSE_X_HUMAN = "ctd_prenet_rtmpose_x_human"

    # DLCRNet variants
    DLCRNET_STRIDE16_MS5 = "dlcrnet_stride16_ms5"
    DLCRNET_STRIDE32_MS5 = "dlcrnet_stride32_ms5"

    # RTMPose variants (top-down)
    RTMPOSE_S = "rtmpose_s"
    RTMPOSE_M = "rtmpose_m"
    RTMPOSE_X = "rtmpose_x"

    # AnimalTokenPose variant (inference only)
    ANIMALTOKENPOSE_BASE = "animaltokenpose_base"


class DetectorType(str, Enum):
    """Enumeration of detector types."""

    SSDLITE = "ssdlite"
    FASTERRCNN_RESNET50_FPN_V2 = "fasterrcnn_resnet50_fpn_v2"
    FASTERRCNN_MOBILENET_V3_LARGE_FPN = "fasterrcnn_mobilenet_v3_large_fpn"


class DatasetType(str, Enum):
    """Enumeration of dataset types."""

    # TODO @deruyter92 2026-02-05: Add other dataset types as needed.
    MULTIANIMAL_IMGAUG = "multi-animal-imgaug"
