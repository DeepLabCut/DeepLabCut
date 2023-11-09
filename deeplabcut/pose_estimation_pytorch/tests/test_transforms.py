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
