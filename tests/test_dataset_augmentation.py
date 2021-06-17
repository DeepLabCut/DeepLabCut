import imgaug.augmenters as iaa
import pytest
from deeplabcut.pose_estimation_tensorflow.dataset import augmentation


@pytest.mark.parametrize(
    "width, height, n_crops",
    [
        (200, 200, 5),
        (300, 300, 10),
        (400, 400, 20),
    ]
)
def test_keypoint_aware_cropping(
    sample_image,
    sample_keypoints,
    width,
    height,
    n_crops,
):
    aug = augmentation.KeypointAwareCropsToFixedSize(
        width=width,
        height=height,
        n_crops=n_crops,
    )
    images_aug, keypoints_aug = aug(
        images=[sample_image],
        keypoints=[sample_keypoints],
    )
    assert len(images_aug) == len(keypoints_aug) == n_crops
    assert all(im.shape[:2] == (height, width) for im in images_aug)
    # Ensure at least a keypoint is visible in each crop
    assert all(len(kpts) for kpts in keypoints_aug)

    # Test passing in a batch of frames
    n_samples = 8
    images_aug, keypoints_aug = aug(
        images=[sample_image] * n_samples,
        keypoints=[sample_keypoints] * n_samples,
    )
    assert len(images_aug) == len(keypoints_aug) == n_crops * n_samples


@pytest.mark.parametrize(
    "width, height, n_crops",
    [
        (200, 200, 5),
        (300, 300, 10),
        (400, 400, 20),
    ]
)
def test_sequential(
    sample_image,
    sample_keypoints,
    width,
    height,
    n_crops,
):
    # Guarantee that images smaller than crop size are handled fine
    very_small_image = sample_image[:50, :50]
    aug = augmentation.Sequential([
        iaa.PadToFixedSize(width, height),
        augmentation.KeypointAwareCropsToFixedSize(
            width, height, n_crops,
        ),
    ])
    images_aug, keypoints_aug = aug(
        images=[very_small_image],
        keypoints=[sample_keypoints],
    )
    assert len(images_aug) == len(keypoints_aug) == n_crops
    assert all(im.shape[:2] == (height, width) for im in images_aug)
    # Ensure at least a keypoint is visible in each crop
    assert all(len(kpts) for kpts in keypoints_aug)
