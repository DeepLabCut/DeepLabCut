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

from deeplabcut.pose_estimation_pytorch.models.detectors.base import DETECTORS


@pytest.mark.parametrize("box_score_thresh, expected", [(None, 0.01), (0.4, 0.4)])
def test_fasterrcnn_null_box_score_thresh_uses_default(box_score_thresh, expected):
    detector = DETECTORS.build(
        {
            "type": "FasterRCNN",
            "variant": "fasterrcnn_mobilenet_v3_large_fpn",
            "pretrained": False,
            "box_score_thresh": box_score_thresh,
        }
    )
    assert detector.model.roi_heads.score_thresh == expected
