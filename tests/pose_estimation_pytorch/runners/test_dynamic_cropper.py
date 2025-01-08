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
"""Tests the dynamic cropper"""
import pytest

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.runners.dynamic_cropping import DynamicCropper


@pytest.mark.parametrize("dynamic", [(False, 0.5, 10)])
def test_build_dynamic_cropper(dynamic: tuple[bool, float, int]):
    cropper = DynamicCropper.build(*dynamic)
    should_be_built, threshold, margin = dynamic
    if should_be_built:
        assert isinstance(cropper, DynamicCropper)
        assert cropper.threshold == threshold
        assert cropper.margin == margin
    else:
        assert cropper is None


@pytest.mark.parametrize("batch_size", [0, 2, 8])
def test_dynamic_fails_with_image_batch(batch_size: int):
    cropper = DynamicCropper(threshold=0.6, margin=10)
    with pytest.raises(RuntimeError):
        cropper.crop(torch.zeros(batch_size, 3, 128, 128))


def test_dynamic_fails_with_variable_frame_size():
    cropper = DynamicCropper(threshold=0.6, margin=10)
    cropper.crop(torch.zeros(1, 3, 64, 64))
    with pytest.raises(RuntimeError):
        cropper.crop(torch.zeros(1, 3, 128, 128))


def test_dynamic_fails_with_update_before_crop():
    cropper = DynamicCropper(threshold=0.6, margin=10)
    with pytest.raises(RuntimeError):
        cropper.update(torch.ones(5, 17, 3))


@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.8])
def test_dynamic_cropper_does_nothing_with_low_quality(threshold: float):
    cropper = DynamicCropper(threshold=threshold, margin=10)
    image_in = torch.ones((1, 3, 32, 32))
    cropper.crop(image_in)
    for i in range(10):
        pose = _generate_random_pose(
            (32, 64),
            min_score=0.0,
            max_score=threshold - 0.001,
            seed=i,
        )
        cropper.update(pose)
        image_out = cropper.crop(image_in)
        assert torch.equal(image_in, image_out)


@pytest.mark.parametrize(
    "pose, threshold, margin, expected_crop",
    [
        ([[float("nan"), float("nan"), float("nan")]], 0.1, 10, [0, 0, 64, 64]),
        ([[float("nan"), 30, 0.0]], 0.5, 10, [0, 0, 64, 64]),
        ([[20, 30, 0.0]], 0.5, 10, [0, 0, 64, 64]),
        ([[20, 30, 0.49]], 0.5, 10, [0, 0, 64, 64]),
        ([[20, 30, 0.8]], 0.5, 10, [10, 20, 30, 40]),
        ([[20, 30, 0.8], [float("nan"), float("nan"), 0.2]], 0.5, 15, [5, 15, 35, 45]),
        ([[20, 30, 0.8], [5, 5, 0.2]], 0.5, 15, [0, 0, 35, 45]),
        ([[20, 30, 0.8], [35, 30, 0.79]], 0.8, 5, [15, 25, 40, 35]),
        ([[40, 10, 0.2], [35, 15, 0.79]], 0.3, 8, [27, 2, 48, 23]),
        (
            [
                [[float("nan"), float("nan"), float("nan")]],
                [[float("nan"), float("nan"), float("nan")]],
            ],
            0.15, 10, [0, 0, 64, 64]
        ),
        (
            [
                [[20, 30, 0.8], [5, 12, 0.2]],
                [[40, 10, 0.2], [35, 15, 0.79]],
            ],
            0.15, 5, [0, 5, 45, 35]
        ),
    ],
)
def test_dynamic_cropper_basic_crop(
    pose: list[list[float]],
    threshold: float,
    margin: int,
    expected_crop: tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = expected_crop
    crop_w, crop_h = x1 - x0, y1 - y0

    image_in = torch.zeros((1, 3, 64, 64))
    image_in[:, :, y0:y1, x0:x1] = 1
    expected_image_out = torch.ones((1, 3, crop_h, crop_w))

    cropper = DynamicCropper(threshold=threshold, margin=margin)
    image_out = cropper.crop(image_in)
    assert torch.equal(image_out, image_in)

    cropper.update(torch.tensor(pose))
    image_out = cropper.crop(image_in)
    assert image_out.shape == expected_image_out.shape
    assert torch.equal(image_out, expected_image_out)

    pose_out = torch.tensor(pose)
    print("\nPose in")
    print(pose_out.numpy())
    pose_out[..., 0] -= x0
    pose_out[..., 1] -= y0
    print("Pose out before update")
    print(pose_out.numpy())
    cropper.update(pose_out)
    print("Pose out after update")
    print(pose_out.numpy())
    np.testing.assert_allclose(pose_out.numpy(), np.array(pose))


def _generate_random_pose(
    image_shape: tuple[int, int],
    min_score: float,
    max_score: float,
    num_animals: int = 3,
    num_keypoints: int = 7,
    seed: int = 0,
) -> torch.Tensor:
    gen = np.random.default_rng(seed)
    pose = gen.random((num_animals, num_keypoints, 3))
    pose[..., 0] *= image_shape[0]
    pose[..., 1] *= image_shape[1]
    pose[..., 2] = (pose[..., 2] * (max_score - min_score)) + min_score
    return torch.from_numpy(pose)
