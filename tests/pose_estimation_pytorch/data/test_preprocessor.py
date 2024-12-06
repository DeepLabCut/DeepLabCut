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

from deeplabcut.pose_estimation_pytorch.data.transforms import build_resize_transforms
from deeplabcut.pose_estimation_pytorch.data.preprocessor import AugmentImage


@pytest.mark.parametrize(
    "data",
    [
        {
            "image_shape": (2, 4, 4),
            "resize_transform": {"height": 5, "width": 4, "keep_ratio": True},
            "output_shape": (2, 4, 4),
            "padded_shape": (5, 4, 4),  # single offset as not a batch
            "output_context": {"offsets": (0, 0), "scales": (1, 1)}
        },
        {
            "image_shape": (1, 2, 4, 4),  # as batch
            "resize_transform": {"height": 10, "width": 4, "keep_ratio": True},
            "output_shape": (1, 2, 4, 4),
            "padded_shape": (1, 10, 4, 4),
            "output_context": {"offsets": [(0, 0)], "scales": [(1, 1)]}
        },
        {
            "image_shape": (2, 4, 3),
            "resize_transform": {"height": 10, "width": 8, "keep_ratio": True},
            "output_shape": (4, 8, 3),
            "padded_shape": (10, 8, 3),
            "output_context": {"offsets": (0, 0), "scales": (0.5, 0.5)}
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
