import albumentations as A
import numpy as np
import os
from torch.utils.data import DataLoader
import pytest
import random

import deeplabcut.pose_estimation_pytorch as dlc
import deeplabcut.utils.auxiliaryfunctions as dlc_auxfun


def _get_dataset(path, transform, mode="train"):
    dlc_project = dlc.DLCProject(path, shuffle=1)
    dataset = dlc.PoseDataset(dlc_project, transform=transform, mode=mode)
    return dataset


def _get_openfield_dataset(transform=None):
    dlc_path = dlc_auxfun.get_deeplabcut_path()
    repo_path = os.path.dirname(dlc_path)
    openfield_path = os.path.join(repo_path, "examples", "openfield-Pranav-2018-10-30")

    return _get_dataset(openfield_path, transform=transform)


@pytest.mark.parametrize("batch_size", [1, 2, random.randint(2, 20)])
def test_iter_all_dataset_no_transform(batch_size):
    if batch_size > 1:  # if batched, all images need to be the same size
        transform = A.Compose(
            [A.Resize(512, 512)],
            keypoint_params=A.KeypointParams(format="xy"),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        )
    else:
        transform = None
    dataset = _get_openfield_dataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    key_set = set(["image", "original_size", "annotations"])
    anno_key_set = set(
        ["keypoints", "area", "ids", "boxes", "image_id", "is_crowd", "labels"]
    )

    max_num_animals = dataset.max_num_animals
    num_keypoints = dataset.num_joints
    for i, item in enumerate(dataloader):
        is_last_batch = i == (len(dataloader) - 1)
        assert (
            set(item.keys()) == key_set
        ), "the key returned don't match the required ones"

        anno = item["annotations"]
        assert (
            set(anno.keys()) == anno_key_set
        ), "the annotation keys returned don't match the required ones"

        assert (len(item["image"].shape) == 4) and (
            (item["image"].shape[:2] == (batch_size, 3)) or is_last_batch
        ), "image shape is not (batch_size, 3, h, w)"

        b, _, h, w = item["image"].shape
        kpts, bboxes = anno["keypoints"], anno["boxes"]
        assert (
            kpts.shape == (batch_size, max_num_animals, num_keypoints, 2)
            or is_last_batch
        ), "keypoints have the wrong shape"
        assert (
            bboxes.shape == (batch_size, max_num_animals, 4) or is_last_batch
        ), "boxes have the wrong shape"
        assert ((bboxes[:, :, 0] + bboxes[:, :, 2]) <= w).all() and (
            (bboxes[:, :, 1] + bboxes[:, :, 3]) <= h
        ).all(), "boxes don't seem to be un the format (x, y, w, h)"


def _generate_random_test_values_aug(min_exa):
    batch_size = random.randint(1, 20)
    x_size = random.randint(50, 600)
    y_size = random.randint(50, 600)
    exageration = random.randint(min_exa, 99)

    return (batch_size, x_size, y_size, exageration)


@pytest.mark.parametrize(
    "batch_size, x_size, y_size, exageration",
    [
        (1, 512, 512, 1),
        _generate_random_test_values_aug(1),
        _generate_random_test_values_aug(50),
    ],
)
def test_iter_all_augmented_dataset(batch_size, x_size, y_size, exageration):
    transform = A.Compose(
        [
            A.Affine(
                scale=(1 - exageration * 0.01, 1 + exageration),
                rotate=(-exageration * 2, exageration * 2),
                translate_px=(-exageration * 10, exageration * 10),
            ),
            A.Resize(y_size, x_size),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    dataset = _get_openfield_dataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    key_set = set(["image", "original_size", "annotations"])
    anno_key_set = set(
        ["keypoints", "area", "ids", "boxes", "image_id", "is_crowd", "labels"]
    )

    max_num_animals = dataset.max_num_animals
    num_keypoints = dataset.num_joints
    for i, item in enumerate(dataloader):
        is_last_batch = i == (len(dataloader) - 1)
        assert (
            set(item.keys()) == key_set
        ), "the key returned don't match the required ones"

        anno = item["annotations"]
        assert (
            set(anno.keys()) == anno_key_set
        ), "the annotation keys returned don't match the required ones"

        assert (len(item["image"].shape) == 4) and (
            (item["image"].shape[:2] == (batch_size, 3)) or is_last_batch
        ), "image shape is not (batch_size, 3, h, w)"

        kpts, bboxes = anno["keypoints"], anno["boxes"]
        b, _, h, w = item["image"].shape
        assert (h == y_size) and (w == x_size)
        assert (
            kpts.shape == (batch_size, max_num_animals, num_keypoints, 2)
            or is_last_batch
        ), "keypoints have the wrong shape"
        assert (
            bboxes.shape == (batch_size, max_num_animals, 4) or is_last_batch
        ), "boxes have the wrong shape"
        assert ((bboxes[:, :, 0] + bboxes[:, :, 2]) <= w).all() and (
            (bboxes[:, :, 1] + bboxes[:, :, 3]) <= h
        ).all()
