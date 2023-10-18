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
from __future__ import annotations

from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset

from deeplabcut.pose_estimation_pytorch.data.utils import (
    _crop_image_keypoints,
    _crop_and_pad_image_torch,
)
from deeplabcut.pose_estimation_pytorch.data.utils import _extract_keypoints_and_bboxes
from deeplabcut.pose_estimation_pytorch.data.utils import apply_transform
from deeplabcut.pose_estimation_pytorch.data.utils import map_id_to_annotations
from deeplabcut.pose_estimation_pytorch.data.utils import map_image_path_to_id
from deeplabcut.pose_estimation_pytorch.data.utils import pad_to_length


@dataclass(frozen=True)
class PoseDatasetParameters:
    """Parameters for a pose dataset

    Attributes:
        bodyparts: the names of bodyparts in the dataset
        unique_bpts: the names of unique bodyparts, or an empty list
        individuals: the names of individuals
        with_center_keypoints: whether to compute center keypoints for individuals
        color_mode: {"RGB", "BGR"} the mode to load images in
    """

    bodyparts: list[str]
    unique_bpts: list[str]
    individuals: list[str]
    with_center_keypoints: bool = False
    color_mode: str = "RGB"
    cropped_image_size: tuple[int, int] | None = None

    @property
    def num_joints(self) -> int:
        return len(self.bodyparts)

    @property
    def num_unique_bpts(self) -> int:
        return len(self.unique_bpts)

    @property
    def max_num_animals(self) -> int:
        return len(self.individuals)


@dataclass
class PoseDataset(Dataset):
    """A pose dataset"""

    images: list[dict[str, str]]
    annotations: list[dict]
    parameters: PoseDatasetParameters
    transform: A.BaseCompose | None = None
    mode: str = "train"
    task: str = "BU"

    def __post_init__(self):
        self.image_path_id_map = map_image_path_to_id(self.images)
        self.annotation_idx_map = map_id_to_annotations(self.annotations)

    def __len__(self):
        # TODO: TD should only return the number of annotations that aren't unique_bodyparts
        return len(self.images) if self.task in ("BU", "DT") else len(self.annotations)

    def _get_raw_item(self, index: int) -> tuple[str, list[dict], int]:
        """
        Retrieve the image path and annotations for the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[str, list]: A tuple containing the image path and annotations.

        Note:
            This method is used by the __getitem__ method to fetch raw data from the dataset storage.
            If `self.crop` is True, it returns the image path and a list with a single annotation.
            Otherwise, it returns the image path and a list of annotations for all instances in the image.
        """
        image_path = self.images[index]["file_name"]
        image_id = self.image_path_id_map[image_path]
        annotations = [
            self.annotations[annotations_id]
            for annotations_id in self.annotation_idx_map[image_id]
        ]
        return image_path, annotations, image_id

    def _get_raw_item_crop(self, index: int) -> tuple[str, list[dict], int]:
        annotations = self.annotations[index]
        image_id = annotations["image_id"]
        annotations = [annotations]

        image_id2path = {
            image_id: image_path
            for (
                image_path,
                image_id,
            ) in self.image_path_id_map.items()
        }

        return image_id2path[image_id], annotations, image_id

    def __getitem__(self, index: int) -> dict:
        """
        Gets the item at the specified index from the dataset.

        Args:
            index: ordered number of the items in the dataset

        Returns:
            dict: corresponding to the image annotations, with keys:
            {
                "image": image tensor of shape (c, h, w),
                "image_id": the ID of the image,
                "path": the filepath to the image,
                "original_size": the original (h, w) size before transforms
                "offsets": the (x, y) offsets to apply to the keypoints in TD mode
                "scales": the (x, y) scales to apply to the keypoints in TD mode
                "annotations": {
                    "keypoints": array of keypoints, invisible keypoints appear as (-1,-1)
                    "keypoints_unique": the unique keypoints, if there are any
                    "area": array of animals area in this image
                    "boxes": the bounding boxes in this image
                    "is_crowd": is_crowd annotations
                    "labels": category_id annotations for boxes
                },
            }
        """
        image_path, annotations, image_id = self._get_data_based_on_task(index)
        image, original_size = self._load_image(image_path)
        (
            keypoints,
            keypoints_unique,
            bboxes,
            annotations_merged,
        ) = self.extract_keypoints_and_bboxes(
            annotations,
            image.shape,
        )
        offsets = np.zeros((self.parameters.max_num_animals, 2))
        scales = (1, 1)
        if self.task == "TD":
            if self.parameters.cropped_image_size is None:
                raise ValueError(
                    "You must specify a cropped image size for top-down models",
                )
            if len(bboxes) > 1:
                raise ValueError(
                    "There can only be one bbox per item in TD datasets, found "
                    f"{bboxes} for {index} (image {image_path})"
                )
            bboxes = bboxes.astype(int)

            # TODO: The following code should be replaced by a numpy version
            image, offsets, scales = _crop_and_pad_image_torch(
                image, bboxes[0], "xywh", self.parameters.cropped_image_size[0]
            )
            keypoints[:, :, 0] = (keypoints[:, :, 0] - offsets[0]) / scales[0]
            keypoints[:, :, 1] = (keypoints[:, :, 1] - offsets[1]) / scales[1]

            # coords = [
            #     (bboxes[0][0], bboxes[0][0] + bboxes[0][2]),  # bboxes=xywh
            #     (bboxes[0][1], bboxes[0][1] + bboxes[0][3]),
            # ]
            # image, keypoints, offsets, scales = self.crop(
            #     image,
            #     keypoints,
            #     coords,
            #     self.parameters.cropped_image_size,
            # )
            bboxes = np.zeros(
                (0, 4)
            )  # No more bounding boxes as we cropped around them

        transformed = self.apply_transform_all_keypoints(
            image,
            keypoints,
            keypoints_unique,
            bboxes,
        )
        keypoints = transformed["keypoints"]

        if self.parameters.with_center_keypoints:
            keypoints = self.add_center_keypoints(keypoints)

        item = self._prepare_final_data_dict(
            transformed["image"],
            keypoints,
            transformed["keypoints_unique"],
            original_size,
            image_path,
            bboxes,
            image_id,
            annotations_merged,
            offsets,
            scales,
        )
        return item

    def _prepare_final_data_dict(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        keypoints_unique: np.ndarray,
        original_size: tuple[int, int],
        image_path: str,
        bboxes: np.array,
        image_id: int,
        annotations_merged: dict,
        offsets: tuple[int, int],
        scales: tuple[float, float],
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        return {
            "image": image.transpose((2, 0, 1)),
            "image_id": image_id,
            "path": image_path,
            "original_size": original_size,
            "offsets": offsets,
            "scales": scales,
            "annotations": self._prepare_final_annotation_dict(
                keypoints,
                keypoints_unique,
                bboxes,
                annotations_merged,
            ),
        }

    def _prepare_final_annotation_dict(
        self,
        keypoints: np.ndarray,
        keypoints_unique: np.ndarray,
        bboxes: np.array,
        annotations_merged: dict,
    ) -> dict[str, np.ndarray]:
        num_animals = self.parameters.max_num_animals
        is_crowd = np.array(annotations_merged["iscrowd"])
        cat_ids = np.array(annotations_merged["category_id"])
        return {
            "keypoints": pad_to_length(keypoints[..., :2], num_animals, -1),
            "keypoints_unique": keypoints_unique[..., :2],
            "area": pad_to_length(annotations_merged["area"], num_animals, 0),
            "boxes": pad_to_length(bboxes, num_animals, 0),
            "is_crowd": pad_to_length(is_crowd, num_animals, 0),
            "labels": pad_to_length(cat_ids, num_animals, -1),
        }

    def _load_image(self, image_path):
        image = cv2.imread(image_path)
        if self.parameters.color_mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image.shape

    def _get_data_based_on_task(self, index: int) -> tuple[str, list[dict], int]:
        """
        Retrieve data based on the specified task.

        For the 'TD' (top-down pose estimation) task:
        - Provides a cropped image and its annotations.
        - The shape of annotations['keypoints'] is (1, num_joints, 2).

        For 'BU' and 'DT' tasks:
        - Provides the full, non-cropped image and its annotations.
        - The shape of annotations['keypoints'] is (max_num_animals, num_joints, 2).

        Args:
            index: Index of the item in the dataset.

        Returns:
            tuple: Tuple containing the image path, annotations, and image ID.
        """
        if self.task == "TD":
            return self._get_raw_item_crop(index)
        elif self.task in ["BU", "DT"]:
            return self._get_raw_item(index)

        raise ValueError(
            f"Unknown task: {self.task}. " 'Task should be one of: "BU", "TD", "DT"',
        )

    def apply_transform_all_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        keypoints_unique: np.ndarray,
        bboxes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Transforms the image using this class's transform

        Args:
            image: the image to transform
            keypoints: an array of shape (num_individuals, num_joints, 3) containing
                the keypoints in the image
            keypoints_unique: an array of shape (num_unique_bodyparts, 3) containing
                the unique keypoints in the image
            bboxes: the bounding boxes in the image

        Returns:
            the augmented image, keypoints and bboxes, in format
            {
                "image": (h, w, c),
                "keypoints": (num_individuals, num_joints, 3),
                "keypoints_unique": (num_unique_bodyparts, 3),
                "bboxes": (4,),
            }
        """
        class_labels = [
            f"individual{i}_{bpt}"
            for i in range(self.parameters.max_num_animals)
            for bpt in self.parameters.bodyparts
        ] + [f"unique_{bpt}" for bpt in self.parameters.unique_bpts]

        all_keypoints = keypoints.reshape(-1, 3)
        if self.parameters.num_unique_bpts > 0:
            all_keypoints = np.concatenate([all_keypoints, keypoints_unique], axis=0)

        transformed = apply_transform(
            self.transform,
            image,
            all_keypoints,
            bboxes,
            class_labels=class_labels,
        )
        if self.parameters.num_unique_bpts > 0:
            keypoints = transformed["keypoints"][
                : -self.parameters.num_unique_bpts
            ].reshape(*keypoints.shape)
            keypoints_unique = transformed["keypoints"][
                -self.parameters.num_unique_bpts :
            ]
            keypoints_unique = keypoints_unique.reshape(
                self.parameters.num_unique_bpts, 3
            )
        else:
            keypoints = transformed["keypoints"].reshape(*keypoints.shape)
            keypoints_unique = np.zeros((0,))

        transformed["keypoints"] = keypoints
        transformed["keypoints_unique"] = keypoints_unique
        return transformed

    @staticmethod
    def crop(
        image: np.ndarray,
        keypoints,
        coords: tuple[tuple[int, int], tuple[int, int]],
        output_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Crop the image based on a given bounding box and resize it to the desired output size.

        Args:
            image: the image to transform
            keypoints: an array of shape (num_individuals, num_joints, 3) containing
                the keypoints in the image
            coords: A bounding box defined as ((x_center, y_center), (width, height)).
            output_size: Desired size for the output cropped, padded and resized image.

        Returns:
            Cropped (and possibly padded) and resized image.
            Offsets used for cropping.
            Padding sizes.
            Scale factor used to resize the image.
        """
        return _crop_image_keypoints(image, keypoints, coords, output_size)

    def extract_keypoints_and_bboxes(
        self,
        annotations: list[dict],
        image_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list]]:
        """
        Args:
            annotations: COCO-style annotations
            image_shape: the (h, w, c) shape of the image for which to get annotations

        Returns:
            keypoints with shape (n_annotation, num_joints, 3)
            unique_keypoints with shape (num_unique_bpts, 3)
            bboxes in xywh format with shape (n_annotation, 4)
            annotations_merged, where each key contains n_annotation values
        """
        return _extract_keypoints_and_bboxes(
            annotations,
            image_shape,
            self.parameters.num_joints,
            self.parameters.num_unique_bpts,
        )

    @staticmethod
    def add_center_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """Adds a keypoint in the mean of each individual"""
        center_keypoints = keypoints.copy()
        center_keypoints[center_keypoints == -1] = np.nan
        center_keypoints = np.nanmean(center_keypoints, axis=1)
        np.nan_to_num(center_keypoints, copy=False, nan=-1)
        return np.concatenate((keypoints, center_keypoints[:, None, :]), axis=1)
