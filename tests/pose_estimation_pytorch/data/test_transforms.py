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
"""Tests the custom transforms"""
import albumentations as A
import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data import transforms


@pytest.mark.parametrize(
    "height, width, image_shapes",
    [
        (200, 200, [(300, 300, 3), (1000, 1000, 3), (1024, 1024, 1)]),
        (512, 512, [(1024, 1024, 3), (128, 128, 4), (300, 300, 1)]),
        (1024, 512, [(600, 300, 3), (4096, 2048, 3), (50, 25, 1)]),
        (800, 1300, [(80, 130, 3), (1600, 2600, 4), (1200, 1950, 1)]),
    ],
)
def test_dlc_resize_pad_good_aspect_ratio(height, width, image_shapes):
    aug = transforms.KeepAspectRatioResize(width=width, height=height, mode="pad")
    for image_shape in image_shapes:
        fake_image = np.zeros(image_shape)
        transformed = aug(image=fake_image, keypoints=[])
        assert transformed["image"].shape[:2] == (height, width)
        assert transformed["image"].shape[2] == fake_image.shape[2]


@pytest.mark.parametrize(
    "data",
    [
        {
            "height": 200,
            "width": 200,
            "in_shapes": [(100, 50, 3), (50, 400, 3)],
            "out_shapes": [(200, 100, 3), (25, 200, 3)],
        },
        {
            "height": 128,
            "width": 256,
            "in_shapes": [(100, 100, 3), (512, 256, 3)],
            "out_shapes": [(128, 128, 3), (128, 64, 3)],
        },
    ],
)
def test_dlc_resize_pad_bad_aspect_ratio(data):
    aug = transforms.KeepAspectRatioResize(width=data["width"], height=data["height"], mode="pad")
    for in_shape, out_shape in zip(data["in_shapes"], data["out_shapes"]):
        fake_image = np.zeros(in_shape)
        transformed = aug(image=fake_image, keypoints=[])
        assert transformed["image"].shape == out_shape


@pytest.mark.parametrize(
    "data",
    [
        {
            "height": 200,
            "width": 200,
            "in_shape": (100, 50, 3),
            "out_shape": (200, 100, 3),
            "in_keypoints": [(50.0, 50.0), (25.0, 10.0)],
            "out_keypoints": [(100.0, 100.0), (50.0, 20.0)],
        },
        {
            "height": 512,
            "width": 256,
            "in_shape": (1024, 1024, 3),
            "out_shape": (256, 256, 3),
            "in_keypoints": [(512.0, 512.0), (100.0, 10.0)],
            "out_keypoints": [(128.0, 128.0), (25.0, 2.5)],
        },
    ],
)
def test_dlc_resize_pad_bad_aspect_ratio_with_keypoints(data):
    aug = transforms.KeepAspectRatioResize(width=data["width"], height=data["height"], mode="pad")
    transform = A.Compose(
        [aug],
        keypoint_params=A.KeypointParams("xy", remove_invisible=False),
    )
    fake_image = np.zeros(data["in_shape"])
    transformed = transform(image=fake_image, keypoints=data["in_keypoints"])
    assert transformed["image"].shape == data["out_shape"]
    assert transformed["keypoints"] == data["out_keypoints"]


def test_coarse_dropout():
    aug = transforms.CoarseDropout(
        max_holes=10,
        max_height=0.05,
        min_height=0.01,
        max_width=0.05,
        min_width=0.01,
        p=0.5,
    )
