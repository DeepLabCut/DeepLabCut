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
"""Configuration for CTD tracking"""
from dataclasses import dataclass


@dataclass
class CTDTrackingConfig:
    """Configuration for CTD tracking.

    Args:
        bu_on_lost_idv: When True, the BU model is run when there are fewer conditions
            found than the expected number of individuals in the video.
        bu_min_frequency: The minimum frequency at which the BU model is run to generate
            conditions. If None, the BU model is only run to initialize the pose in the
            first frame, and then is not run again. If a positive number N, the BU model
            is run every N frames. The BU predictions are then combined with the CTD
            predictions to continue the tracklets.
        bu_max_frequency: The maximum frequency at which the BU model can be run. Must
            be greater than `bu_min_frequency`. When there are fewer conditions than
            individuals expected in the video and `bu_on_lost_idv` is True, the BU model
            may be run on every frame. This can happen if individuals can disappear from
            the video, and each frame may have a variable number of individuals. If
            `bu_max_frequency` is set to N, then the BU model will be run at most every
            N-th frame, which improves the inference speed of the model.
        threshold_bu_add: The OKS threshold below which a BU pose must be (wrt. any
            existing CTD pose) to be added to the poses.
        threshold_ctd: The score threshold below which detected keypoints are NOT given
            to the CTD model to predict pose for the next frame.
        threshold_nms: The OKS threshold for non-maximum suppression to remove
            duplicates poses when two CTD model predictions converge to a single animal.
    """
    bu_on_lost_idv: bool = True
    bu_min_frequency: int | None = None
    bu_max_frequency: int | None = 100
    threshold_bu_add: float = 0.25
    threshold_ctd: float = 0.01
    threshold_nms: float = 0.9

    @staticmethod
    def build(config: dict, video_fps: float | None = None) -> "CTDTrackingConfig":
        """Builds a CTD tracking configuration from a configuration dictionary.

        Examples:
            Building a CTDTrackingConfig from a basic dict:
            >>> ctd_tracking = CTDTrackingConfig.build(
            >>>   dict(bu_on_lost_idv=True, threshold_nms=0.75)
            >>> )

            Building a CTDTrackingConfig from a basic dict:
            >>> ctd_tracking = CTDTrackingConfig.build(
            >>>   dict(
            >>>     bu_on_lost_idv=True,
            >>>     bu_max_frequency=5,    # When no FPS is given, this is in frames!
            >>>     threshold_nms=0.5,
            >>>   )
            >>> )

            Building a CTDTrackingConfig from a dict for a video with a given FPS:
            >>> ctd_tracking = CTDTrackingConfig.build(
            >>>   dict(
            >>>     bu_on_lost_idv=True,
            >>>     bu_min_frequency=1,    # When an FPS is given, this is in seconds!
            >>>     bu_max_frequency=5,    # When an FPS is given, this is in seconds!
            >>>     threshold_ctd=0.1,
            >>>     threshold_nms=0.9
            >>>   ),
            >>>   video_fps=30.0,
            >>> )
        """
        kwargs = {**config}
        if video_fps is not None:
            if "bu_min_frequency" in config:
                kwargs["bu_min_frequency"] = int(config["bu_min_frequency"] * video_fps)
            if "bu_max_frequency" in config:
                kwargs["bu_max_frequency"] = int(config["bu_max_frequency"] * video_fps)
        return CTDTrackingConfig(**kwargs)
