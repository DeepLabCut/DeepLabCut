import imgaug.augmenters as iaa
import pytest
from deeplabcut.pose_estimation_tensorflow.datasets import augmentation


@pytest.mark.parametrize(
    "width, height",
    [
        (200, 200),
        (300, 300),
        (400, 400),
    ]
)
def test_keypoint_aware_cropping(
    sample_image,
    sample_keypoints,
    width,
    height,
):
    aug = augmentation.KeypointAwareCropToFixedSize(
        width=width,
        height=height,
    )
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
    ]
)
def test_sequential(
    sample_image,
    sample_keypoints,
    width,
    height,
):
    # Guarantee that images smaller than crop size are handled fine
    very_small_image = sample_image[:50, :50]
    aug = iaa.Sequential([
        iaa.PadToFixedSize(width, height),
        augmentation.KeypointAwareCropToFixedSize(width, height),
    ])
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
