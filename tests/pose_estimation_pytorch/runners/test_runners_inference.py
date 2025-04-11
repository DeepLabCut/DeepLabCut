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
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import deeplabcut.pose_estimation_pytorch.data.postprocessor as post
import deeplabcut.pose_estimation_pytorch.data.preprocessor as prep
import deeplabcut.pose_estimation_pytorch.runners.inference as inference
from deeplabcut.pose_estimation_pytorch import get_load_weights_only
from deeplabcut.pose_estimation_pytorch.task import Task


@patch("deeplabcut.pose_estimation_pytorch.runners.train.build_optimizer", Mock())
@pytest.mark.parametrize("task", [Task.DETECT, Task.TOP_DOWN, Task.BOTTOM_UP])
@pytest.mark.parametrize("weights_only", [None, True, False])
def test_load_weights_only_with_build_training_runner(task: Task, weights_only: bool):
    with patch("deeplabcut.pose_estimation_pytorch.runners.base.torch.load") as load:
        snapshot = "snapshot.pt"
        runner = inference.build_inference_runner(
            task=task,
            model=Mock(),
            device="cpu",
            snapshot_path=snapshot,
            load_weights_only=weights_only,
        )
        if weights_only is None:
            weights_only = get_load_weights_only()
        load.assert_called_once_with(
            snapshot, map_location="cpu", weights_only=weights_only
        )


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
            {"mock": {"index": i[0, 0, 0].detach().numpy()}} for i in inputs
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
        [4, 23, 5, 20, 64, 100],
    ],
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


def test_dynamic_pose_inference_calls_dynamic():
    pose_batch = torch.zeros((1, 1, 1, 3))
    pose_batch_updated = torch.ones((1, 1, 1, 3))

    image_crop = Mock()
    image_crop.__len__ = Mock(return_value=1)

    model = Mock()
    model.get_predictions = Mock()
    model.get_predictions.return_value = dict(bodypart=dict(poses=pose_batch))

    dynamic = Mock()
    dynamic.crop = Mock()
    dynamic.crop.return_value = image_crop
    dynamic.update = Mock()
    dynamic.update.return_value = pose_batch_updated

    runner = inference.PoseInferenceRunner(
        model=model,
        dynamic=dynamic,
        batch_size=1,
    )
    image = torch.zeros((1, 3, 64, 64))
    updated_pose = runner.predict(image)
    dynamic.crop.assert_called_once_with(image)
    dynamic.update.assert_called_once_with(pose_batch)

    assert len(updated_pose) == 1
    np.testing.assert_allclose(
        updated_pose[0]["bodypart"]["poses"], pose_batch_updated[0].cpu().numpy(),
    )


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
