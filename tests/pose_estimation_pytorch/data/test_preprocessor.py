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
"""Tests the pre-processors"""
import albumentations as A
import numpy as np
import pytest
from albumentations import BaseCompose

from deeplabcut.pose_estimation_pytorch.data.transforms import build_resize_transforms
from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
    AugmentImage,
    build_conditional_top_down_preprocessor,
)


@pytest.mark.parametrize(
    "data",
    [
        {
            "image_shape": (2, 4, 4),
            "resize_transform": {"height": 5, "width": 4, "keep_ratio": True},
            "output_shape": (2, 4, 4),
            "padded_shape": (5, 4, 4),  # single offset as not a batch
            "output_context": {"offsets": (0, 0), "scales": (1, 1)},
        },
        {
            "image_shape": (1, 2, 4, 4),  # as batch
            "resize_transform": {"height": 10, "width": 4, "keep_ratio": True},
            "output_shape": (1, 2, 4, 4),
            "padded_shape": (1, 10, 4, 4),
            "output_context": {"offsets": [(0, 0)], "scales": [(1, 1)]},
        },
        {
            "image_shape": (2, 4, 3),
            "resize_transform": {"height": 10, "width": 8, "keep_ratio": True},
            "output_shape": (4, 8, 3),
            "padded_shape": (10, 8, 3),
            "output_context": {"offsets": (0, 0), "scales": (0.5, 0.5)},
        },
    ],
)
def test_augment_image_rescaling(data):
    resize_transform = build_resize_transforms(data["resize_transform"])
    transform = A.Compose(
        resize_transform,
        keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    preprocessor = AugmentImage(transform)
    img = np.ones(data["image_shape"])
    transformed_image, context = preprocessor(img, context={})
    print()
    print(transformed_image[:, :, 0])  # first channel
    print(context)
    assert np.sum(transformed_image) == np.sum(np.ones(data["output_shape"]))
    assert context == data["output_context"]
    assert transformed_image.shape == data["padded_shape"]


ctd_preprocessor = build_conditional_top_down_preprocessor(
    color_mode="RGB",
    transform=A.Compose(
        build_resize_transforms({"height": 100, "width": 100, "keep_ratio": True}),
        keypoint_params=A.KeypointParams("xy", remove_invisible=False),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    ),
    bbox_margin=0,
    top_down_crop_size=(256, 256),
)


@pytest.mark.parametrize(
    "data",
    [
        # two well-defined individuals
        {
            "image_shape": (100, 100, 3),
            "context": {
                "cond_kpts": np.array(
                    [[[10, 10, 0.8], [20, 20, 0.8]], [[60, 60, 0.8], [70, 70, 0.8]]]
                )
            },
            "output_context": {
                "cond_kpts": np.array(
                    [[[10, 10, 0.8], [20, 20, 0.8]], [[60, 60, 0.8], [70, 70, 0.8]]]
                ),
                "bboxes": [np.array([10, 10, 10, 10]), np.array([60, 60, 10, 10])],
                "offsets": [(10, 10), (60, 60)],
                "scales": [(0.1, 0.1), (0.1, 0.1)],
            },
        },
        # one individual has 0 keypoints
        {
            "image_shape": (100, 100, 3),
            "context": {
                "cond_kpts": np.array(
                    [[[10, 10, 0.8], [20, 20, 0.8]], [[60, 60, 0.0], [70, 70, 0.0]]]
                )
            },
            "output_context": {
                "cond_kpts": np.array(
                    [
                        [[10, 10, 0.8], [20, 20, 0.8]],
                    ]
                ),
                "bboxes": [np.array([10, 10, 10, 10])],
                "offsets": [(10, 10)],
                "scales": [(0.1, 0.1)],
            },
        },
        # one individual has only 1 keypoints
        {
            "image_shape": (100, 100, 3),
            "context": {
                "cond_kpts": np.array(
                    [[[10, 10, 0.8], [20, 20, 0.8]], [[60, 60, 0.0], [70, 70, 0.9]]]
                )
            },
            "output_context": {
                "cond_kpts": np.array(
                    [
                        [[10, 10, 0.8], [20, 20, 0.8]],
                    ]
                ),
                "bboxes": [np.array([10, 10, 10, 10])],
                "offsets": [(10, 10)],
                "scales": [(0.1, 0.1)],
            },
        },
        # two individuals but one is low confidence
        {
            "image_shape": (100, 100, 3),
            "context": {
                "cond_kpts": np.array(
                    [[[10, 10, 0.8], [20, 20, 0.8]], [[60, 60, 0.01], [70, 70, 0.01]]]
                )
            },
            "output_context": {
                "cond_kpts": np.array(
                    [
                        [[10, 10, 0.8], [20, 20, 0.8]],
                    ]
                ),
                "bboxes": [np.array([10, 10, 10, 10])],
                "offsets": [(10, 10)],
                "scales": [(0.1, 0.1)],
            },
        },
    ],
)
def test_conditional_top_down_preprocessor(data):
    input_img = np.ones(data["image_shape"])

    output_img, output_context = ctd_preprocessor(input_img, context=data["context"])

    for context_key in ["cond_kpts", "bboxes", "offsets", "scales"]:
        assert deep_equal(
            output_context[context_key], data["output_context"][context_key]
        )


def deep_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    else:
        return a == b
