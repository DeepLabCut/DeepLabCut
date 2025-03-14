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
"""Tests data utils"""
import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch.data.utils as utils


@pytest.mark.parametrize(
    "keypoints, expected_bboxes, params",
    [
        (
            [[0, 0, 2], [10, 5, 2]],
            [0, 0, 10, 5],
            dict(image_w=1024, image_h=1024, margin=0),
        ),
        (
            [[-1, -1, 2], [3, 4, 2]],
            [0, 0, 3, 4],
            dict(image_w=1024, image_h=1024, margin=0),
        ),
        (
            [[0, 0, 2], [10, 5, 2]],
            [0, 0, 5, 3],
            dict(image_w=5, image_h=3, margin=0),
        ),
        (
            [[0, 0, 2], [10, 5, 2]],
            [0, 0, 5, 3],
            dict(image_w=5, image_h=3, margin=10),
        ),
        (
            [[[0, 0, 2], [10, 5, 2]]],
            [[0, 0, 10, 5]],
            dict(image_w=1024, image_h=1024, margin=0),
        ),
        (
            [
                [[4, 1, 2], [10, 5, 2], [3, 12, 0]],
                [[7, 3, 2], [2, 0, -1], [1, 12, 2]],
            ],
            [
                [4, 1, 6, 4],
                [1, 3, 6, 9],
            ],
            dict(image_w=1024, image_h=1024, margin=0),
        ),
        (
            [
                [[4, 1, 2], [10, 5, 2], [3, 12, 0]],
                [[7, 3, 2], [2, 0, -1], [1, 12, 2]],
            ],
            [
                [2, 0, 10, 7],
                [0, 1, 9, 13],
            ],
            dict(image_w=1024, image_h=1024, margin=2),
        ),
        (
            [
                [[4, 1, 2], [10, 5, 2], [3, 12, 0]],
                [[7, 3, 2], [2, 0, -1], [1, 12, 2]],
            ],
            [
                [2, 0, 8, 7],
                [0, 1, 9, 9],
            ],
            dict(image_w=10, image_h=10, margin=2),
        ),
        (
            [
                [[4, 1, 2], [10, 5, 2], [3, 12, 0]],
                [[7, 3, 0], [2, 0, -1], [1, 12, 0]],
            ],
            [
                [2, 0, 8, 7],
                [0, 0, 0, 0],
            ],
            dict(image_w=10, image_h=10, margin=2),
        ),
    ],
)
def test_bbox_from_keypoints(keypoints, expected_bboxes, params):
    keypoints = np.asarray(keypoints, dtype=float)
    bboxes = utils.bbox_from_keypoints(keypoints, **params)
    expected_bboxes = np.asarray(expected_bboxes, dtype=float)
    np.testing.assert_array_almost_equal(bboxes, expected_bboxes)
