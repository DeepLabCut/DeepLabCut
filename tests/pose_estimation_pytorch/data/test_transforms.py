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
import random

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


@pytest.mark.parametrize(
    "data",
    [
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=10.0,
                shift_prob=0.0,
                scale_factor=[0.1, 2.0],
                scale_prob=0.0,
            ),
        },
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.0,
                shift_prob=1.0,
                scale_factor=[1.0, 1.0],
                scale_prob=1.0,
                sampling="uniform",  # truncnorm throws an error if delta is 0
            ),
        },
    ],
)
def test_random_bbox_transform_does_not_modify_with_base_config(data: dict) -> None:
    _set_random_seed()
    h, w, c = data["image_shape"]

    # generate 100 bboxes
    bboxes = _gen_random_bboxes(np.random.default_rng(seed=0), 100, w, h)

    t = A.Compose(
        [transforms.RandomBBoxTransform(**data["transform_config"])],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    output = t(
        image=np.zeros((h, w, c)), bboxes=bboxes, bbox_labels=np.zeros(len(bboxes)),
    )
    print("Output bounding boxes")
    for out_bbox in output["bboxes"]:
        print(out_bbox)
    print()
    bboxes_out = np.asarray(output["bboxes"])
    print("bboxes")
    print(bboxes_out)
    print()
    np.testing.assert_array_almost_equal(bboxes, bboxes_out)


@pytest.mark.parametrize(
    "data",
    [
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.0,
                shift_prob=0.0,
                scale_factor=[0.25, 0.5],
                scale_prob=1.0,
            ),
        },
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.0,
                shift_prob=0.0,
                scale_factor=[1.0, 1.5],
                scale_prob=1.0,
            ),
        },
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.0,
                shift_prob=0.0,
                scale_factor=[0.5, 1.25],
                scale_prob=1.0,
            ),
        },
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.0,
                shift_prob=0.0,
                scale_factor=[0.5, 1.5],
                scale_prob=0.5,
            ),
        },
    ],
)
def test_random_bbox_transform_scale(data: dict) -> None:
    _set_random_seed()
    h, w, c = data["image_shape"]

    # generate 100 bboxes
    bboxes = _gen_random_bboxes(np.random.default_rng(seed=0), 100, w, h)

    t = A.Compose(
        [transforms.RandomBBoxTransform(**data["transform_config"])],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    output = t(
        image=np.zeros((h, w, c)), bboxes=bboxes, bbox_labels=np.zeros(len(bboxes)),
    )
    print("Output bounding boxes")
    for out_bbox in output["bboxes"]:
        print(out_bbox)
    print()

    bboxes_out = np.asarray(output["bboxes"])
    scale_low, scale_high = data["transform_config"]["scale_factor"]
    for bbox_in_wh, bbox_out_wh in zip(bboxes[:, 2:], bboxes_out[:, 2:]):
        print("bbox_in_wh", bbox_in_wh)
        w, h = bbox_in_wh[0].item(), bbox_in_wh[1].item()
        w_low, w_high = w * scale_low, w * scale_high
        h_low, h_high = h * scale_low, h * scale_high
        print("(w, w_low, w_high)", w, w_low, w_high)
        print("(h, h_low, h_high)", h, h_low, h_high)
        assert w_low <= bbox_out_wh[0].item() <= w_high
        assert h_low <= bbox_out_wh[1].item() <= h_high


@pytest.mark.parametrize(
    "data",
    [
        {
            "image_shape": [480, 640, 3],
            "transform_config": dict(
                shift_factor=0.1,
                shift_prob=1.0,
                scale_factor=[1.0, 1.0],
                scale_prob=0.0,
            ),
        },
    ],
)
def test_random_bbox_transform_shift(data: dict) -> None:
    _set_random_seed()
    h, w, c = data["image_shape"]

    # generate 100 bboxes
    bboxes = _gen_random_bboxes(np.random.default_rng(seed=0), 100, w, h)

    t = A.Compose(
        [transforms.RandomBBoxTransform(**data["transform_config"])],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    output = t(
        image=np.zeros((h, w, c)), bboxes=bboxes, bbox_labels=np.zeros(len(bboxes)),
    )
    print("Output bounding boxes")
    for out_bbox in output["bboxes"]:
        print(out_bbox)
    print()

    bboxes_out = np.asarray(output["bboxes"])
    shift = data["transform_config"]["shift_factor"]
    for bbox_in, bbox_out in zip(bboxes, bboxes_out):
        print("bbox_in", bbox_in)
        x, y, w, h = bbox_in
        x_out, y_out, w_out, h_out = bbox_out
        max_shift_x, max_shift_y = w * shift, h * shift
        assert x - max_shift_x <= x_out <= x + max_shift_x
        assert y - max_shift_y <= y_out <= y + max_shift_y


def _set_random_seed():
    np.random.seed(0)
    random.seed(0)


def _gen_random_bboxes(
    gen: np.random.Generator, num_bboxes: int, w: int, h: int,
) -> np.ndarray:
    image_wh = np.array([w, h])
    bboxes = np.zeros((num_bboxes, 4))
    # sample x, y in the images
    bboxes[:, :2] = image_wh * gen.random((num_bboxes, 2))
    # sample w, h with the space remaining
    bboxes[:, 2:] = (image_wh - bboxes[:, :2]) * gen.random((num_bboxes, 2))

    print()
    print("Input bounding boxes")
    print(bboxes)
    return bboxes
