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
"""Generated detector configs store ``box_score_thresh: null``; building a
Faster R-CNN from such a config must not leave torchvision's score threshold
as None, which crashes ``postprocess_detections`` during inference."""

import pytest

from deeplabcut.pose_estimation_pytorch.config.model import DetectorModelConfig
from deeplabcut.pose_estimation_pytorch.models.detectors.base import DETECTORS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.01),
        (0.0, 0.0),
        (0.4, 0.4),
    ],
)
def test_detector_box_score_thresh(value, expected):
    config = DetectorModelConfig(box_score_thresh=value)

    assert config.box_score_thresh == expected


def test_detector_box_score_thresh_default():
    config = DetectorModelConfig()

    assert config.box_score_thresh == 0.01


def test_build_from_config_with_null_threshold():
    """The detector must not be built with torchvision's score_thresh set to None.

    Nothing at the model layer normalizes the threshold any more, so this asserts
    the property the config layer is relied upon for: a legacy ``null`` never
    reaches the torchvision constructor.
    """
    config = DetectorModelConfig.from_dict(
        {
            "type": "FasterRCNN",
            "variant": "fasterrcnn_mobilenet_v3_large_fpn",
            "box_score_thresh": None,
        }
    )

    detector = DETECTORS.build(config)

    assert detector.model.roi_heads.score_thresh == 0.01
