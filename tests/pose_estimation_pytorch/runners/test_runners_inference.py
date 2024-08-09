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
"""Tests inference runners"""
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.data.postprocessor as post
import deeplabcut.pose_estimation_pytorch.data.preprocessor as prep
import deeplabcut.pose_estimation_pytorch.runners.inference as inference


class MockInferenceRunner(inference.InferenceRunner):
    """Mocks the predict function for an inference runner"""

    def __init__(
        self,
        batch_size: int = 1,
        preprocessor: prep.Preprocessor | None = None,
        postprocessor: post.Postprocessor | None = None,
    ) -> None:
        super().__init__(
            model=Mock(),
            batch_size=batch_size,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        self.batch_shapes = []

    def predict(self, inputs: torch.Tensor) -> list[dict[str, dict[str, np.ndarray]]]:
        self.batch_shapes.append(tuple(inputs.shape))
        return [  # return first elem of input
            {"mock": {"index": i[0, 0, 0].detach().numpy()}}
            for i in inputs
        ]


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_mock_bottom_up(batch_size):
    h, w = 640, 480
    images = [i * np.ones((1, 3, h, w)) for i in range(10)]

    runner = MockInferenceRunner(batch_size=batch_size)
    predictions = runner.inference(images)

    print()
    print(f"Num images: {len(predictions)}")
    print(f"Num predictions: {len(predictions)}")
    print(f"Batch shapes: {runner.batch_shapes}")
    print(80 * "-")
    for i in images:
        print(i[0, 0, 0, 0])
        print("----")
    print(80 * "-")
    for p in predictions:
        print(p)
        print("----")

    _check_batch_shapes(batch_size, h, w, runner.batch_shapes)
    assert len(images) == len(predictions)
    for i, p in zip(images, predictions):
        assert len(p) == 1  # only 1 output per image
        assert i[0, 0, 0, 0] == p[0]["mock"]["index"]


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "detections_per_image",
    [
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1],  # some frames might not have predictions
        [0, 0, 0, 5, 2],
        [1, 2, 3, 4],
        [3, 4, 2, 1, 4],
        [4, 23, 5, 20, 64, 100]
    ]
)
def test_mock_top_down(batch_size, detections_per_image):
    h, w = 8, 8
    images = []
    for index, num_detections in enumerate(detections_per_image):
        if num_detections == 0:
            detections = np.zeros((0, 3, 1, 1))  # random shape when no detections
        else:
            detections = np.concatenate(
                [
                    (1_000_000 * (index + 1) + i) * np.ones((1, 3, h, w))
                    for i in range(num_detections)
                ],
                axis=0,
            )

        images.append(detections)

    runner = MockInferenceRunner(batch_size=batch_size)
    predictions = runner.inference(images)

    print()
    print(f"Num images: {len(predictions)}")
    print(f"Num predictions: {len(predictions)}")
    print(80 * "-")
    for i in images:
        for i_det in i:
            print(i_det.shape)
            print(i_det[0, 0, 0])
        print("----")

    print(80 * "-")
    for p in predictions:
        print(p)
        print("----")

    _check_batch_shapes(batch_size, h, w, runner.batch_shapes)

    assert len(images) == len(predictions)
    for i, p in zip(images, predictions):
        assert len(p) == len(i)  # one prediction per input
        for i_det, p_det in zip(i, p):
            print(i_det.shape)
            print(p_det["mock"]["index"])
            assert i_det[0, 0, 0] == p_det["mock"]["index"]


def _check_batch_shapes(batch_size, h, w, batch_shapes) -> None:
    for b in batch_shapes[:-1]:
        assert b[0] == batch_size
        assert b[1] == 3
        assert b[2] == h
        assert b[3] == w

    assert batch_shapes[-1][0] <= batch_size
    assert batch_shapes[-1][1] <= 3
    assert batch_shapes[-1][2] <= h
    assert batch_shapes[-1][3] <= w
