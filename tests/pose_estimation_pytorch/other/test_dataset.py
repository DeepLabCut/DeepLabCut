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
import os
import random
from pathlib import Path
from unittest.mock import Mock, patch

import albumentations as A
import pytest
from torch.utils.data import DataLoader

import deeplabcut.pose_estimation_pytorch as dlc
import deeplabcut.utils.auxiliaryfunctions as dlc_auxfun
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset import create_training_dataset


def mock_config() -> Mock:
    aux_functions = Mock()
    aux_functions.read_config_as_dict = Mock()
    aux_functions.read_config_as_dict.return_value = {
        "data": {"train": {}, "inference": {}},
        "metadata": {
            "project_path": "",
            "pose_config_path": "",
            "bodyparts": ["snout", "leftear", "rightear", "tailbase"],
            "unique_bodyparts": [],
            "individuals": ["animal"],
            "with_identity": False,
        },
        "method": "bu",
    }
    return aux_functions


@patch("deeplabcut.pose_estimation_pytorch.data.base.config", mock_config())
def _get_dataset(path, transform, mode="train"):
    project_root = Path(path)
    if not (project_root / "training-datasets").exists():
        print(str(project_root / "config.yaml"))
        create_training_dataset(
            config=str(project_root / "config.yaml"),
            net_type="resnet_50",
            engine=Engine.PYTORCH,
        )

    loader = dlc.DLCLoader(Path(project_root) / "config.yaml", shuffle=1)
    dataset = loader.create_dataset(transform=transform, mode=mode)
    return dataset


def _get_openfield_dataset(transform=None):
    dlc_path = dlc_auxfun.get_deeplabcut_path()
    repo_path = os.path.dirname(dlc_path)
    openfield_path = os.path.join(repo_path, "examples", "openfield-Pranav-2018-10-30")

    return _get_dataset(openfield_path, transform=transform)


key_set = {
    "offsets",
    "path",
    "scales",
    "image",
    "original_size",
    "annotations",
    "image_id",
    "context",
}
anno_key_set = {
    "keypoints",
    "keypoints_unique",
    "with_center_keypoints",
    "area",
    "boxes",
    "is_crowd",
    "labels",
    "individual_ids",
}


@pytest.mark.parametrize("batch_size", [1, 2, random.randint(2, 20)])
def test_iter_all_dataset_no_transform(batch_size):
    if batch_size > 1:  # if batched, all images need to be the same size
        transform = A.Compose(
            [A.Resize(512, 512)],
            keypoint_params=A.KeypointParams(format="xy"),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        )
    else:
        transform = A.Compose(
            [A.Normalize()],
            keypoint_params=A.KeypointParams(format="xy"),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
        )
    dataset = _get_openfield_dataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    max_num_animals = dataset.parameters.max_num_animals
    num_keypoints = dataset.parameters.num_joints
    for i, item in enumerate(dataloader):
        is_last_batch = i == (len(dataloader) - 1)
        assert (
            set(item.keys()) == key_set
        ), f"the key returned don't match the required ones: {item.keys()} != {key_set}"

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
            kpts.shape == (batch_size, max_num_animals, num_keypoints, 3)
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
    exaggeration = random.randint(min_exa, 99)

    return batch_size, x_size, y_size, exaggeration


@pytest.mark.parametrize(
    "batch_size, x_size, y_size, exaggeration",
    [
        (1, 512, 512, 1),
        _generate_random_test_values_aug(1),
        _generate_random_test_values_aug(50),
    ],
)
def test_iter_all_augmented_dataset(batch_size, x_size, y_size, exaggeration):
    transform = A.Compose(
        [
            A.Affine(
                scale=(1 - exaggeration * 0.01, 1 + exaggeration),
                rotate=(-exaggeration * 2, exaggeration * 2),
                translate_px=(-exaggeration * 10, exaggeration * 10),
            ),
            A.Resize(y_size, x_size),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]),
    )
    dataset = _get_openfield_dataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    max_num_animals = dataset.parameters.max_num_animals
    num_keypoints = dataset.parameters.num_joints
    for i, item in enumerate(dataloader):
        is_last_batch = i == (len(dataloader) - 1)
        assert (
            set(item.keys()) == key_set
        ), f"the key returned don't match the required ones: {item.keys()} != {key_set}"

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
            kpts.shape == (batch_size, max_num_animals, num_keypoints, 3)
            or is_last_batch
        ), "keypoints have the wrong shape"
        assert (
            bboxes.shape == (batch_size, max_num_animals, 4) or is_last_batch
        ), "boxes have the wrong shape"
        assert ((bboxes[:, :, 0] + bboxes[:, :, 2]) <= w).all() and (
            (bboxes[:, :, 1] + bboxes[:, :, 3]) <= h
        ).all()
