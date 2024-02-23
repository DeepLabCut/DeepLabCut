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
import numpy as np
import pytest

from deeplabcut.pose_estimation_pytorch.data import transforms


@pytest.mark.parametrize("width, height", [(200, 200), (300, 300), (400, 400)])
def test_keypoint_aware_cropping(width, height):
    fake_image = np.empty((600, 600, 3))
    fake_keypoints = [(i * 100, i * 100, 0, 0) for i in range(1, 6)]
    aug = transforms.KeypointAwareCrop(
        width=width, height=height, crop_sampling="density"
    )
    transformed = aug(image=fake_image, keypoints=fake_keypoints)
    assert transformed["image"].shape[:2] == (height, width)
    # Ensure at least a keypoint is visible in each crop
    assert len(transformed["keypoints"])


def test_grayscale():
    fake_image = np.ones((600, 600, 3))
    fake_image *= np.random.uniform(0, 255, size=fake_image.shape)
    fake_image = fake_image.astype(np.uint8)
    gray = transforms.Grayscale(alpha=1, p=1)
    aug_image = gray(image=fake_image)["image"]
    assert aug_image.shape == fake_image.shape

    gray = transforms.Grayscale(alpha=0, p=1)
    aug_image = gray(image=fake_image)["image"]
    assert np.allclose(fake_image, aug_image)

    with pytest.warns(UserWarning, match="clipped"):
        gray = transforms.Grayscale(alpha=1.5)
    assert gray.alpha == 1


def test_coarse_dropout():
    fake_image = np.ones((300, 300, 3))
    fake_image *= np.random.uniform(0, 255, size=fake_image.shape)
    fake_image = fake_image.astype(np.uint8)
    cd = transforms.CoarseDropout(max_height=0.9999, max_width=0.9999, p=1)
    kpts = np.random.rand(10, 2) * 300
    aug_kpts = cd(image=fake_image, keypoints=kpts)["keypoints"]
    assert len(aug_kpts) == kpts.shape[0]
    assert np.isnan([c for kpt in aug_kpts for c in kpt]).all()
