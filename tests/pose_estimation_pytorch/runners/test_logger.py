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
"""Tests loggers"""
from typing import Any

import pytest
import torch

import deeplabcut.pose_estimation_pytorch.runners.logger as logging


class MockImageLogger(logging.ImageLoggerMixin):
    """Mock image logger"""

    def log_images(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        targets: dict[str, dict[str, torch.Tensor]],
        step: int,
    ) -> None:
        pass


@pytest.mark.parametrize(
    "keypoints",
    [
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[float("nan"), float("nan")], [float("nan"), float("nan")]],
        ],
        [
            [[0.0, 0.0], [1, 1], [2, 2]],
        ],
        [[[float("nan"), 0.0], [1, 1], [2, 2]]],
        [[[-1.0, -1.0], [1, 1], [2, 2]]],
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
        ],
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
    ],
)
@pytest.mark.parametrize("denormalize", [True, False])
def test_prepare_image(keypoints: list[list[float]], denormalize: bool) -> None:
    print()
    print(torch.tensor(keypoints).shape)
    print()

    logger = MockImageLogger()
    logger._prepare_image(
        image=torch.zeros((3, 256, 256)),
        denormalize=denormalize,
        keypoints=torch.tensor(keypoints),
        bboxes=None,
    )
