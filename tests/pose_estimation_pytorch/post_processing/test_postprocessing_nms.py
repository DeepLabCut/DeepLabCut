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
"""Tests pose NMS"""
import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.post_processing.nms as nms


@pytest.mark.parametrize(
    "poses, score_threshold, expected_kept",
    [
        (
            [
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            0.1,
            [True],  # a single pose should be kept
        ),
        (
            [
                [[0.0, np.nan, 0], [0, 0, 0], [0, 0, 0]],
            ],
            0.1,
            [True],  # a single pose should be kept
        ),
        (
            [
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            0.1,
            [False, False],  # no valid poses
        ),
        (
            [
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            0.1,
            [False, True, False, False],  # a single valid pose
        ),
        (
            [
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
                [[100.0, 100, 0.89], [110, 110, 0.89], [120, 120, 0.89]],
            ],
            0.1,
            [True, True],  # two valid poses, far apart
        ),
        (
            [
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
                [[100.0, 100, 0.8], [110, 110, 0.8], [120, 120, 0.8]],
            ],
            0.1,
            [False, True, True],  # two valid poses, far apart
        ),
        (
            [
                [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[100.0, 100, 0.8], [110, 110, 0.8], [120, 120, 0.8]],
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
            ],
            0.1,
            [False, True, True],  # two valid poses, far apart, sorted by score
        ),
        (
            [
                [[0.0, 0, 0.89], [10, 10, 0.89], [20, 20, 0.89]],
                [[100.0, 100, 0.8], [110, 110, 0.8], [120, 120, 0.8]],
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
            ],
            0.1,
            [False, True, True],  # two valid poses, far apart, sorted by score, one suppressed
        ),
        (
            [
                [[1.0, 0, 0.89], [11, 10, 0.89], [21, 20, 0.89]],
                [[100.0, 100, 0.8], [110, 110, 0.8], [120, 120, 0.8]],
                [[0.0, 0, 0.9], [10, 10, 0.9], [20, 20, 0.9]],
            ],
            0.1,
            [False, True, True],  # two valid poses, far apart, sorted by score, one suppressed
        ),
    ]
)
def test_oks_nms_post_processing(poses, score_threshold, expected_kept):
    """Tests pose NMS"""
    kept = nms.nms_oks(
        predictions=np.asarray(poses),
        oks_threshold=0.9,
        oks_sigmas=0.1,
        score_threshold=0.1,
    )
    assert kept.tolist() == expected_kept
