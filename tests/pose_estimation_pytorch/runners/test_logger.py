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
from pathlib import Path
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
    image = torch.ones((3, 256, 256))
    keypoints = torch.tensor(keypoints)

    print()
    print(f"IMAGE: {image.shape}")
    print(f"KEYPOINTS: {keypoints.shape}")
    for k in keypoints:
        print(k)
    print()
    print()

    logger = MockImageLogger()
    logger._prepare_image(
        image=image,
        denormalize=denormalize,
        keypoints=keypoints,
        bboxes=None,
    )


def test_csv_logger_resume(tmp_path: Path) -> None:
    """Test CSVLogger preserves data when resuming from snapshot"""
    log_file = tmp_path / "learning_stats.csv"
    
    # Initial training: log some metrics
    logger1 = logging.CSVLogger(str(tmp_path), "learning_stats.csv")
    logger1.log({"loss": 0.5, "accuracy": 0.8}, step=1)
    logger1.log({"loss": 0.4, "accuracy": 0.9}, step=2)
    
    assert log_file.exists()
    assert len(logger1._steps) == 2
    
    # Resume training: should load existing data
    logger2 = logging.CSVLogger(str(tmp_path), "learning_stats.csv")
    assert len(logger2._steps) == 2
    assert logger2._steps == [1, 2]
    assert logger2._metric_store[0]["loss"] == 0.5
    assert logger2._metric_store[1]["accuracy"] == 0.9
    
    # Log new data: should append, not overwrite
    logger2.log({"loss": 0.3, "accuracy": 0.95}, step=3)
    assert len(logger2._steps) == 3
    assert logger2._steps == [1, 2, 3]
