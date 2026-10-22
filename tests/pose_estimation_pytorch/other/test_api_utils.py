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

import deeplabcut.pose_estimation_pytorch.data.transforms as transforms

TRANSFORM_DICTS = {
    "auto-padding": {"auto_padding": {"pad_height_divisor": 64, "pad_width_divisor": 27}},
    "resize": {"resize": {"height": 512, "width": 256, "keep_ratio": True}},
    "typical-augmentations": {
        "covering": True,
        "gaussian_noise": 12.75,
        "hist_eq": True,
        "motion_blur": True,
        "normalize_images": True,
        "rotation": 30,
        "scale_jitter": [0.5, 1.25],
        "auto_padding": {"pad_width_divisor": 64, "pad_height_divisor": 27},
    },
    "extreme-augmentations": {
        "covering": True,
        "gaussian_noise": 100,
        "hist_eq": True,
        "motion_blur": True,
        "normalize_images": True,
        "rotation": 180,
        "scale_jitter": [0.03, 20],
        "auto_padding": {"pad_width_divisor": 64, "pad_height_divisor": 27},
    },
}

# The seed is per case, so each transform config is exercised with
# different images and poses
TRANSFORM_CASES = [
    pytest.param(transform_dict, seed, id=case_id)
    for seed, (case_id, transform_dict) in enumerate(TRANSFORM_DICTS.items())
]


@pytest.mark.parametrize("transform_dict, seed", TRANSFORM_CASES)
def test_build_transforms(transform_dict, seed):
    rng = np.random.default_rng(seed)

    w, h = rng.integers(100, 1001, size=2)
    num_keypoints = int(rng.integers(1, 101))
    num_animals = int(rng.integers(1, 101))

    transform_bbox_aug = transforms.build_transforms(transform_dict)

    for _ in range(10):
        test_image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

        bboxes = rng.integers(
            0,
            min(w - 1, h - 1),
            size=(num_animals, 4),
        )
        bboxes[:, 2] = w - bboxes[:, 0]
        bboxes[:, 3] = h - bboxes[:, 1]

        keypoints = rng.integers(
            0,
            min(w, h),
            size=(num_keypoints, 2),
        )

        with pytest.raises(ValueError):
            transform_bbox_aug(image=test_image)

        with pytest.raises(ValueError):
            transform_bbox_aug(
                image=test_image,
                bboxes=bboxes.copy(),
            )

        with pytest.raises(ValueError):
            transform_bbox_aug(
                image=test_image,
                keypoints=keypoints.copy(),
                bboxes=bboxes.copy(),
            )

        transformed = transform_bbox_aug(
            image=test_image,
            keypoints=keypoints.copy(),
            bboxes=bboxes.copy(),
            bbox_labels=np.arange(num_animals),
            class_labels=[0] * len(keypoints),
        )

        if "resize" in transform_dict:
            assert transformed["image"].shape[:2] == (
                transform_dict["resize"]["height"],
                transform_dict["resize"]["width"],
            )

        if "auto_padding" in transform_dict:
            modh = transform_dict["auto_padding"]["pad_height_divisor"]
            modw = transform_dict["auto_padding"]["pad_width_divisor"]

            assert transformed["image"].shape[0] % modh == 0
            assert transformed["image"].shape[1] % modw == 0

        assert len(transformed["keypoints"]) == len(keypoints)
