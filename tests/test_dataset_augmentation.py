#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import imgaug.augmenters as iaa
import numpy as np
import pytest
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation


@pytest.mark.parametrize(
    "width, height",
    [
        (200, 200),
        (300, 300),
        (400, 400),
    ],
)
def test_keypoint_aware_cropping(
    sample_image,
    sample_keypoints,
    width,
    height,
):
    aug = augmentation.KeypointAwareCropToFixedSize(width=width, height=height)
    images_aug, keypoints_aug = aug(
        images=[sample_image],
        keypoints=[sample_keypoints],
    )
    assert len(images_aug) == len(keypoints_aug) == 1
    assert all(im.shape[:2] == (height, width) for im in images_aug)
    # Ensure at least a keypoint is visible in each crop
    assert all(len(kpts) for kpts in keypoints_aug)

    # Test passing in a batch of frames
    n_samples = 8
    images_aug, keypoints_aug = aug(
        images=[sample_image] * n_samples,
        keypoints=[sample_keypoints] * n_samples,
    )
    assert len(images_aug) == len(keypoints_aug) == n_samples


@pytest.mark.parametrize(
    "width, height",
    [
        (200, 200),
        (300, 300),
        (400, 400),
    ],
)
def test_sequential(
    sample_image,
    sample_keypoints,
    width,
    height,
):
    # Guarantee that images smaller than crop size are handled fine
    very_small_image = sample_image[:50, :50]
    aug = iaa.Sequential(
        [
            iaa.PadToFixedSize(width, height),
            augmentation.KeypointAwareCropToFixedSize(width, height),
        ]
    )
    images_aug, keypoints_aug = aug(
        images=[very_small_image],
        keypoints=[sample_keypoints],
    )
    assert len(images_aug) == len(keypoints_aug) == 1
    assert all(im.shape[:2] == (height, width) for im in images_aug)
    # Ensure at least a keypoint is visible in each crop
    assert all(len(kpts) for kpts in keypoints_aug)

    # Test passing in a batch of frames
    n_samples = 8
    images_aug, keypoints_aug = aug(
        images=[very_small_image] * n_samples,
        keypoints=[sample_keypoints] * n_samples,
    )
    assert len(images_aug) == len(keypoints_aug) == n_samples


def test_keypoint_horizontal_flip(
    sample_image,
    sample_keypoints,
):
    keypoints_flipped = sample_keypoints.copy()
    keypoints_flipped[:, 0] = sample_image.shape[1] - keypoints_flipped[:, 0]
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    aug = augmentation.KeypointFliplr(
        keypoints=list(map(str, range(12))),
        symmetric_pairs=pairs,
    )
    keypoints_aug = aug(
        images=[sample_image],
        keypoints=[sample_keypoints],
    )[
        1
    ][0]
    temp = keypoints_aug.reshape((3, 12, 2))
    for pair in pairs:
        temp[:, pair] = temp[:, pair[::-1]]
    keypoints_unaug = temp.reshape((-1, 2))
    np.testing.assert_allclose(keypoints_unaug, keypoints_flipped)


def test_keypoint_horizontal_flip_with_nans(
    sample_image,
    sample_keypoints,
):
    sample_keypoints[::12] = np.nan
    sample_keypoints[2::12] = np.nan
    keypoints_flipped = sample_keypoints.copy()
    keypoints_flipped[:, 0] = sample_image.shape[1] - keypoints_flipped[:, 0]
    pairs = [(0, 1), (2, 3)]
    aug = augmentation.KeypointFliplr(
        keypoints=list(map(str, range(12))),
        symmetric_pairs=pairs,
    )
    keypoints_aug = aug(
        images=[sample_image],
        keypoints=[sample_keypoints],
    )[
        1
    ][0]
    temp = keypoints_aug.reshape((3, 12, 2))
    for pair in pairs:
        temp[:, pair] = temp[:, pair[::-1]]
    keypoints_unaug = temp.reshape((-1, 2))
    np.testing.assert_allclose(keypoints_unaug, keypoints_flipped)
