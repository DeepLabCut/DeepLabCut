"""Tests the pre-processors"""
import albumentations as A
import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.apis.utils import build_resize_transforms
from deeplabcut.pose_estimation_pytorch.data.preprocessor import AugmentImage
from deeplabcut.pose_estimation_pytorch.data.postprocessor import RescaleAndOffset


@pytest.mark.parametrize(
    "data",
    [
        {
            "predictions": [[[0, 0, 0.95], [20, 30, 0.5]]],
            "offsets": [(0, 0)],
            "scales": [(1, 1)],
            "rescaled": [[[0, 0, 0.95], [20, 30, 0.5]]],
        },
        {
            "predictions": [
                [[0, 0, 0.12], [1000, 0, 0.5]],  # individual 1
                [[18, 2, 0.24], [0, 1000, 0.6]],  # individual 2
            ],
            "offsets": [(0, 0), (0, 0)],
            "scales": [(1, 1), (0.5, 1.0)],
            "rescaled": [
                [[0, 0, 0.12], [1000, 0, 0.5]],  # individual 1
                [[9, 2, 0.24], [0, 1000, 0.6]],  # individual 2
            ],
        },
        {
            "predictions": [
                [[0, 0, 0.95], [20, 30, 0.5]],  # individual 1
                [[110, 5, 0.95], [60, 1200, 0.5]],  # individual 2
            ],
            "offsets": [(12, 5), (27, 10)],
            "scales": [(0.5, 0.5), (0.2, 0.2)],
            "rescaled": [
                [[12, 5, 0.95], [22, 20, 0.5]],  # individual 1
                [[49, 11, 0.95], [39, 250, 0.5]],  # individual 2
            ],
        },
    ],
)
def test_rescale_topdown(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bodyparts"],
        mode=RescaleAndOffset.Mode.KEYPOINT_TD,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bodyparts": np.array(data["predictions"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bodyparts"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bodyparts"], np.array(data["rescaled"]))


@pytest.mark.parametrize(
    "data",
    [
        {
            "predictions": [[[0, 0, 0.95], [20, 30, 0.5]]],
            "offsets": (0, 0),
            "scales": (1, 1),
            "rescaled": [[[0, 0, 0.95], [20, 30, 0.5]]],
        },
        {
            "predictions": [
                [[0, 0, 0.12], [10, 0, 0.5]],  # individual 1
                [[1000, 500, 0.24], [50, 250, 0.6]],  # individual 2
            ],
            "offsets": (5, 7),
            "scales": (0.2, 0.5),
            "rescaled": [
                [[5, 7, 0.12], [7, 7, 0.5]],  # individual 1
                [[205, 257, 0.24], [15, 132, 0.6]],  # individual 2
            ],
        },
    ],
)
def test_rescale_bottom_up(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bodyparts"],
        mode=RescaleAndOffset.Mode.KEYPOINT,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bodyparts": np.array(data["predictions"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bodyparts"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bodyparts"], np.array(data["rescaled"]))


@pytest.mark.parametrize(
    "data",
    [
        {
            "bboxes": [[222.0, 562.0, 721.0, 637.0]],
            "offsets": (0, 0),
            "scales": (1, 1),
            "rescaled": [[222.0, 562.0, 721.0, 637.0]],
        },
        {
            "bboxes": [[386.71875, 219.53125, 281.640625, 248.828125]],
            "offsets": (-768, 0),
            "scales": (2.56, 2.56),
            "rescaled": [[222.0, 562.0, 721.0, 637.0]],
        },
        {
            "bboxes": [
                [0, 0, 100, 100],
                [5, 10, 100, 100],
                [5, 10, 10, 20],
            ],
            "offsets": (3, 7),
            "scales": (2, 0.5),
            "rescaled": [
                [3, 7, 200, 50],
                [13, 12, 200, 50],
                [13, 12, 20, 10],
            ],
        },
    ],
)
def test_rescale_detector(data):
    """expects x_processed = x * scale + offset"""
    postprocessor = RescaleAndOffset(
        keys_to_rescale=["bboxes"],
        mode=RescaleAndOffset.Mode.BBOX_XYWH,
    )
    context = {"scales": data["scales"], "offsets": data["offsets"]}
    predictions = {"bboxes": np.array(data["bboxes"])}
    predictions, context = postprocessor(predictions, context=context)
    print(predictions["bboxes"].tolist())
    print(data["rescaled"])
    np.testing.assert_array_equal(predictions["bboxes"], np.array(data["rescaled"]))
