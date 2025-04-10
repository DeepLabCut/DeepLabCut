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
from __future__ import annotations

from dataclasses import dataclass

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from deeplabcut.pose_estimation_pytorch.data.generative_sampling import (
    GenerativeSampler,
    GenSamplingConfig,
)
from deeplabcut.pose_estimation_pytorch.data.image import load_image, top_down_crop
from deeplabcut.pose_estimation_pytorch.data.utils import (
    _crop_image_keypoints,
    _extract_keypoints_and_bboxes,
    apply_transform,
    bbox_from_keypoints,
    calc_bbox_overlap,
    map_id_to_annotations,
    map_image_path_to_id,
    out_of_bounds_keypoints,
    pad_to_length,
    safe_stack,
)
from deeplabcut.pose_estimation_pytorch.task import Task


@dataclass(frozen=True)
class PoseDatasetParameters:
    """Parameters for a pose dataset

    Attributes:
        bodyparts: the names of bodyparts in the dataset
        unique_bpts: the names of unique bodyparts, or an empty list
        individuals: the names of individuals
        with_center_keypoints: whether to compute center keypoints for individuals
        color_mode: {"RGB", "BGR"} the mode to load images in
        ctd_config: for CTD models, the configuration for bbox calculation and error sampling
        top_down_crop_size: for top-down models, the (width, height) to crop bboxes to
        top_down_crop_margin: for top-down models, the margin to add around bboxes
    """

    bodyparts: list[str]
    unique_bpts: list[str]
    individuals: list[str]
    with_center_keypoints: bool = False
    color_mode: str = "RGB"
    ctd_config: GenSamplingConfig | None = None
    top_down_crop_size: tuple[int, int] | None = None
    top_down_crop_margin: int | None = None
    top_down_crop_with_context: bool = True

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

    images: list[dict]
    annotations: list[dict]
    parameters: PoseDatasetParameters
    transform: A.BaseCompose | None = None
    mode: str = "train"
    task: Task = Task.BOTTOM_UP
    ctd_config: GenSamplingConfig | None = None

    def __post_init__(self):
        self.image_path_id_map = map_image_path_to_id(self.images)
        self.annotation_idx_map = map_id_to_annotations(self.annotations)
        self.img_id_to_index = {
            img["id"]: index for index, img in enumerate(self.images)
        }
        if self.task == Task.TOP_DOWN and (
            self.parameters.top_down_crop_size is None
            or self.parameters.top_down_crop_margin is None
        ):
            raise ValueError(
                "You must specify a ``top_down_crop_size`` and ``top_down_crop_margin``"
                "in your PoseDatasetParameters when the task is TOP_DOWN."
            )

        self.td_crop_size = self.parameters.top_down_crop_size
        self.td_crop_margin = self.parameters.top_down_crop_margin

        if self.task == Task.COND_TOP_DOWN:
            if self.ctd_config is None:
                raise ValueError(
                    "Must specify a ``ctd_config`` in your PoseDatasetParameters for "
                    "CTD models."
                )

            self.generative_sampler = GenerativeSampler(
                self.parameters.num_joints,
                **self.ctd_config.to_dict(),
            )

    def __len__(self):
        # TODO: TD/CTD should only return the number of annotations that aren't unique_bodyparts
        if self.task in (Task.BOTTOM_UP, Task.DETECT):
            return len(self.images)

        return len(self.annotations)

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
        img = self.images[index]
        anns = [self.annotations[idx] for idx in self.annotation_idx_map[img["id"]]]
        return img["file_name"], anns, img["id"]

    def _get_raw_item_crop(self, index: int) -> tuple[str, list[dict], int]:
        ann = self.annotations[index]
        img = self.images[self.img_id_to_index[ann["image_id"]]]
        return img["file_name"], [ann], img["id"]

    def _get_raw_item_crop_context(self, index: int) -> tuple[str, list[dict], int]:
        """
        Includes keypoints from other individuals in the image ("context").
        """
        ann = self.annotations[index]
        img = self.images[self.img_id_to_index[ann["image_id"]]]
        near_anns = []
        for idx in self.annotation_idx_map[img["id"]]:
            # we consider near annotations to be those whose bounding boxes overlap with
            # the current item
            # HACK: add same annotation as near keypoints so that we don't have empty list
            if calc_bbox_overlap(ann["bbox"], self.annotations[idx]["bbox"]) > 0:
                near_anns.append(self.annotations[idx])
        return img["file_name"], [ann] + near_anns, img["id"]

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
        image_path, anns, image_id = self._get_data_based_on_task(index)
        image = load_image(image_path, color_mode=self.parameters.color_mode)
        original_size = image.shape
        (
            keypoints,
            keypoints_unique,
            bboxes,
            annotations_merged,
        ) = self.extract_keypoints_and_bboxes(anns, image.shape)

        # this is applying data augmentations before the cropping
        # though normalization should be applied after the cropping
        transformed = self.apply_transform_all_keypoints(
            image, keypoints, keypoints_unique, bboxes
        )
        image = transformed["image"]
        keypoints = transformed["keypoints"]
        keypoints_unique = transformed["keypoints_unique"]
        bboxes = transformed["bboxes"]
        offsets = (0, 0)
        scales = (1.0, 1.0)

        if self.task in (Task.TOP_DOWN, Task.COND_TOP_DOWN):
            if self.parameters.top_down_crop_size is None:
                raise ValueError(
                    "You must specify a cropped image size for top-down models"
                )
            if len(bboxes) > 1 and self.task == Task.TOP_DOWN:
                raise ValueError(
                    "There can only be one bbox per item in TD datasets, found "
                    f"{bboxes} for {index} (image {image_path})"
                )
            bboxes = bboxes.astype(int)

            if self.task == Task.COND_TOP_DOWN:
                near_keypoints = keypoints[1:]
                keypoints = keypoints[:1]
                synthesized_keypoints = self.generative_sampler(
                    keypoints=keypoints.reshape(-1, 3),
                    near_keypoints=near_keypoints.reshape(len(near_keypoints), -1, 3),
                    area=bboxes[0, 2] * bboxes[0, 3],
                    image_size=original_size,
                )

                # if conditional keypoints are empty, we take original bbox
                if np.any(synthesized_keypoints[..., -1] > 0):
                    bboxes[0] = bbox_from_keypoints(
                        synthesized_keypoints,
                        original_size[0],
                        original_size[1],
                        self.ctd_config.bbox_margin,
                    )

            if bboxes[0, 2] == 0 or bboxes[0, 3] == 0:
                # bbox was augmented out of the image; blank image, no keypoints
                keypoints[..., 2] = 0.0
                if self.task == Task.COND_TOP_DOWN:
                    keypoints = safe_stack(
                        [keypoints, keypoints],
                        (2, 1, self.parameters.num_joints, 3),
                    )

                image = np.zeros(
                    (self.td_crop_size[1], self.td_crop_size[0], image.shape[-1]),
                    dtype=image.dtype,
                )
            else:
                image, offsets, scales = top_down_crop(
                    image,
                    bboxes[0],
                    self.parameters.top_down_crop_size,
                    self.parameters.top_down_crop_margin,
                    crop_with_context=self.parameters.top_down_crop_with_context,
                )

                keypoints[:, :, 0] = (keypoints[:, :, 0] - offsets[0]) / scales[0]
                keypoints[:, :, 1] = (keypoints[:, :, 1] - offsets[1]) / scales[1]
                if self.task == Task.COND_TOP_DOWN:
                    synthesized_keypoints[:, 0] = (
                        synthesized_keypoints[:, 0] - offsets[0]
                    ) / scales[0]
                    synthesized_keypoints[:, 1] = (
                        synthesized_keypoints[:, 1] - offsets[1]
                    ) / scales[1]
                    keypoints = safe_stack(
                        [keypoints, synthesized_keypoints[None, ...]],
                        (2, 1, self.parameters.num_joints, 3),
                    )

                bboxes = bboxes[:1]
                bboxes[..., 0] = (bboxes[..., 0] - offsets[0]) / scales[0]
                bboxes[..., 1] = (bboxes[..., 1] - offsets[1]) / scales[1]
                bboxes[..., 2] = bboxes[..., 2] / scales[0]
                bboxes[..., 3] = bboxes[..., 3] / scales[1]
                bboxes = np.clip(
                    bboxes, 0, self.parameters.top_down_crop_size[0] - 1
                )  # TODO: clip based on [x,y,x,y]?

                # RandomBBoxTransform may move keypoints outside the cropped image
                oob_mask = out_of_bounds_keypoints(keypoints, self.td_crop_size)
                if np.sum(oob_mask) > 0:
                    keypoints[oob_mask, 2] = 0.0

        if self.parameters.with_center_keypoints:
            keypoints = self.add_center_keypoints(keypoints)

        return self._prepare_final_data_dict(
            image,
            keypoints,
            keypoints_unique,
            original_size,
            image_path,
            bboxes,
            image_id,
            annotations_merged,
            offsets,
            scales,
        )

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
        context = dict()
        if self.task == Task.COND_TOP_DOWN:
            context["cond_keypoints"] = keypoints[1, :, :, :].astype(np.single)

        return {
            "image": image.transpose((2, 0, 1)),
            "image_id": image_id,
            "path": image_path,
            "original_size": np.array(original_size),
            "offsets": np.array(offsets, dtype=int),
            "scales": np.array(scales, dtype=float),
            "annotations": self._prepare_final_annotation_dict(
                keypoints, keypoints_unique, bboxes, annotations_merged
            ),
            "context": context,
        }

    def _prepare_final_annotation_dict(
        self,
        keypoints: np.ndarray,
        keypoints_unique: np.ndarray,
        bboxes: np.array,
        anns: dict,
    ) -> dict[str, np.ndarray]:
        num_animals = self.parameters.max_num_animals
        if self.task in (Task.TOP_DOWN, Task.COND_TOP_DOWN):
            num_animals = 1

        bbox_widths = np.maximum(1, bboxes[..., 2])
        bbox_heights = np.maximum(1, bboxes[..., 3])
        area = bbox_widths * bbox_heights
        if "individual_id" not in anns:
            anns["individual_id"] = -np.ones(len(anns["category_id"]), dtype=int)

        individual_ids = anns["individual_id"]
        is_crowd = anns["iscrowd"]
        labels = anns["category_id"]
        if self.task == Task.COND_TOP_DOWN:
            keypoints = keypoints[0]
            area = area[:1]
            bboxes = bboxes[:1]
            individual_ids = individual_ids[:1]
            is_crowd = is_crowd[:1]
            labels = labels[:1]

        # we use ..., :3 to pass the visibility flag along
        return {
            "keypoints": pad_to_length(keypoints[..., :3], num_animals, 0).astype(
                np.single
            ),
            "keypoints_unique": keypoints_unique[..., :3].astype(np.single),
            "with_center_keypoints": self.parameters.with_center_keypoints,
            "area": pad_to_length(area, num_animals, 0).astype(np.single),
            "boxes": pad_to_length(bboxes, num_animals, 0).astype(np.single),
            "is_crowd": pad_to_length(is_crowd, num_animals, 0).astype(int),
            "labels": pad_to_length(labels, num_animals, -1).astype(int),
            "individual_ids": pad_to_length(individual_ids, num_animals, -1).astype(
                int
            ),
        }

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
        if self.task == Task.TOP_DOWN:
            return self._get_raw_item_crop(index)
        elif self.task == Task.COND_TOP_DOWN:
            return self._get_raw_item_crop_context(index)
        elif self.task in (Task.BOTTOM_UP, Task.DETECT):
            return self._get_raw_item(index)

        raise ValueError(f"Unknown task: {self.task}")

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
            for i in range(len(keypoints))
            for bpt in self.parameters.bodyparts
        ] + [f"unique_{bpt}" for bpt in self.parameters.unique_bpts]

        all_keypoints = keypoints.reshape(-1, 3)
        if self.parameters.num_unique_bpts > 0:
            all_keypoints = np.concatenate([all_keypoints, keypoints_unique], axis=0)

        transformed = apply_transform(
            self.transform, image, all_keypoints, bboxes, class_labels=class_labels
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
        transformed["bboxes"] = np.array(transformed["bboxes"])
        if len(transformed["bboxes"]) == 0:
            transformed["bboxes"] = np.zeros((0, 4))

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
        self, anns: list[dict], image_shape: tuple[int, int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Args:
            anns: COCO-style annotations
            image_shape: the (h, w, c) shape of the image for which to get annotations

        Returns:
            keypoints with shape (n_annotation, num_joints, 3)
            unique_keypoints with shape (num_unique_bpts, 3)
            bboxes in xywh format with shape (n_annotation, 4)
            annotations_merged, where each key contains n_annotation values
        """
        return _extract_keypoints_and_bboxes(
            anns,
            image_shape,
            self.parameters.num_joints,
            self.parameters.num_unique_bpts,
        )

    @staticmethod
    def add_center_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """Adds a keypoint in the mean of each individual

        Args:
            keypoints: shape (num_idv, num_kpts, 3)

        Returns:
            keypoints with centers, of shape (num_idv, num_kpts + 1, 3)
        """
        num_idv = keypoints.shape[0]
        centers = np.full((num_idv, 1, 3), np.nan)

        keypoints_xy = keypoints.copy()[..., :2]
        keypoints_xy[keypoints[..., 2] <= 0] = np.nan

        # only set centers for individuals where at least 1 bodypart is visible
        vis_mask = (~np.isnan(keypoints_xy) > 0).all(axis=2).any(axis=1)
        if np.any(vis_mask):
            centers[vis_mask, 0, :2] = np.nanmean(keypoints_xy[vis_mask], axis=1)

        masked_centers = np.any(np.isnan(centers[:, 0, :2]), axis=1)
        centers[masked_centers, 0, 2] = 0
        centers[~masked_centers, 0, 2] = 2
        np.nan_to_num(centers, copy=False, nan=0)

        return np.concatenate((keypoints, centers), axis=1)
