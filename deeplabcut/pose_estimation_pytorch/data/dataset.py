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

import os

import albumentations as A
import cv2
import numpy as np
import torch
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils.auxiliaryfunctions import get_model_folder, read_plainconfig
from torch.utils.data import Dataset

from .base import BaseDataset
from .dlcproject import DLCProject


class PoseDataset(Dataset, BaseDataset):
    """
    Dataset for pose estimation
    """

    def __init__(
        self, project: DLCProject, transform: object = None, mode: str = "train"
    ):
        super().__init__()
        self.transform = transform
        self.project = project
        self.cfg = self.project.cfg

        self.bodyparts = auxiliaryfunctions.get_bodyparts(self.cfg)
        self.num_joints = len(self.bodyparts)

        self.shuffle = self.project.shuffle
        self.project.convert2dict(mode)
        self.dataframe = self.project.dataframe

        modelfolder = os.path.join(
            self.project.proj_root,
            get_model_folder(
                self.cfg["TrainingFraction"][0],
                self.shuffle,
                self.cfg,
                "",
            ),
        )
        pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
        pytorch_cfg = read_plainconfig(pytorch_config_path)
        self.with_center = pytorch_cfg.get("with_center", False)
        self.individuals = self.cfg.get("individuals", ["single"])
        self.individual_to_idx = {}
        for i, indiv in enumerate(self.individuals):
            self.individual_to_idx[indiv] = i
        self.max_num_animals = len(self.individuals)
        self.color_mode = pytorch_cfg.get("colormode", "RGB")

        self.length = self.dataframe.shape[0]
        assert self.length == len(self.project.image_path2image_id.keys())

    def __len__(self):
        return self.length

    def _calc_area_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        w = keypoints[:, :, 0].max(axis=1) - keypoints[:, :, 0].min(axis=1)
        h = keypoints[:, :, 1].max(axis=1) - keypoints[:, :, 1].min(axis=1)
        return w * h
    
    def _keypoint_in_boundary(self, keypoint: list, shape: tuple):
        return (
            (keypoint[0] > 0)
            and (keypoint[1] > 0)
            and (keypoint[0] < shape[1])
            and (keypoint[1] < shape[0])
        )

    def __getitem__(self, index: int) -> dict:
        # load images
        try:
            image_file = self.dataframe.index[index]
            if isinstance(image_file, tuple):
                image_file = os.path.join(self.cfg["project_path"], *image_file)
            else:
                image_file = os.path.join(self.cfg["project_path"], image_file)
        except:
            print(len(self.project.images))
            print(index)

        image = cv2.imread(image_file)
        if self.color_mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape

        # load annotations
        image_id = self.project.image_path2image_id[image_file]
        n_annotations = len(self.project.id2annotations_idx[image_id])

        bodyparts = [bpt for bpt in self.bodyparts]
        if not self.with_center:
            keypoints = np.zeros((self.max_num_animals, self.num_joints, 3))
            num_keypoints_returned = self.num_joints
        else:
            keypoints = np.zeros((self.max_num_animals, self.num_joints + 1, 3))
            num_keypoints_returned = self.num_joints + 1
            bodyparts += ["_center_"]

        bbox_list = []
        bbox_label_list = []
        labels = np.zeros((self.max_num_animals), dtype=np.int64)
        is_crowd = np.zeros((self.max_num_animals), dtype=np.int64)
        ids = np.full((self.max_num_animals), -1, dtype=np.int64)
        image_id = index
        for i, annotation_idx in enumerate(self.project.id2annotations_idx[image_id]):
            _annotation = self.project.annotations[annotation_idx]
            _keypoints, _undef_ids = self.project.annotation2keypoints(_annotation)
            _keypoints = np.array(_keypoints)

            ids[i] = self.individual_to_idx[_annotation["individual"]]

            if self.with_center:
                keypoints[i, :-1, :2] = _keypoints
                keypoints[i, :-1, 2] = _undef_ids

            else:
                keypoints[i, :, :2] = _keypoints
                keypoints[i, :, 2] = _undef_ids

            # If bbox has width and height > 0, add
            annotation_bbox = np.array(_annotation["bbox"])
            if 0 < annotation_bbox[2] and 0 < annotation_bbox[3]:
                bbox_list.append(annotation_bbox)
                bbox_label_list.append(i)

            is_crowd[i] = _annotation["iscrowd"]
            labels[i] = _annotation["category_id"]

        if len(bbox_list) > 0:
            h, w, _ = image.shape
            bboxes = np.stack(bbox_list, axis=0)
            bbox_labels = np.array(bbox_label_list)

            # Sometimes bbox coords are larger than the image because of the margin
            bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
            bboxes[:, 2] = np.clip(np.minimum(bboxes[:, 2], w - bboxes[:, 0]), 0, None)
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h) - np.spacing(0.)
            bboxes[:, 3] = np.clip(np.minimum(bboxes[:, 3], h - bboxes[:, 1]), 0, None)
        else:
            bboxes = np.zeros((0, 4))
            bbox_labels = np.zeros((0,))

        # Needs two be 2 dimensional for albumentations
        keypoints = keypoints.reshape((-1, 3))

        if self.transform:
            class_labels = [
                f"individual{i}_{bpt}"
                for i in range(self.max_num_animals)
                for bpt in bodyparts
            ]
            transformed = self.transform(
                image=image,
                keypoints=keypoints[:, :2],
                bboxes=bboxes,
                class_labels=class_labels,
                bbox_labels=bbox_labels,
            )
            bboxes = transformed["bboxes"]

            # Discard keypoints that aren't in the frame anymore
            shape_transformed = transformed["image"].shape
            transformed["keypoints"] = [
                keypoint
                if self._keypoint_in_boundary(keypoint, shape_transformed)
                else (-1, -1)
                for keypoint in transformed["keypoints"]
            ]

            # Discard keypoints that are undefined
            undef_class_labels = [
                class_labels[i] for i, kpt in enumerate(keypoints) if kpt[2] == 0
            ]
            for label in undef_class_labels:
                new_index = transformed["class_labels"].index(label)
                transformed["keypoints"][new_index] = (-1, -1)

        else:
            transformed = {
                "keypoints": keypoints[:, :2],
                "image": image,
            }

        image = torch.tensor(transformed["image"], dtype=torch.float).permute(
            2, 0, 1
        )  # channels first

        assert len(transformed["keypoints"]) == len(keypoints)
        keypoints = (
            np.array(transformed["keypoints"])
            .reshape((n_annotations, num_keypoints_returned, 2))
            .astype(float)
        )

        # Pad bboxes and labels to always have shape (num_animals, 4)
        #  If we want the original index of the bboxes, we can use the bbox labels
        bbox_tensor = torch.tensor(bboxes, dtype=torch.float)
        if len(bbox_tensor) < self.max_num_animals:
            missing_animals = self.max_num_animals - len(bbox_tensor)
            bbox_tensor = torch.cat(
                [bbox_tensor, torch.zeros((missing_animals, 4))],
                dim=0,
            )

        # TODO Quite ugly
        #
        # Center keypoint needs to be computed after transformation because
        # it should depend on visible keypoints only (which may change after augmentation)
        if self.with_center:
            try:
                keypoints[:, -1, :] = (
                    keypoints[:, :-1, :][~np.any(keypoints[:, :-1, :] == -1, axis=2)]
                    .reshape(n_annotations, -1, 2)
                    .mean(axis=1)
                )
            except ValueError:
                # For at least one annotation every keypoint is out of the frame
                for i in range(keypoints.shape[0]):
                    try:
                        keypoints[i, -1, :] = keypoints[i, :-1, :][
                            ~np.any(keypoints[i, :-1, :] == -1, axis=1)
                        ].mean(axis=0)
                    except ValueError:
                        keypoints[i, -1, :] = np.array([-1, -1])

        np.nan_to_num(keypoints, copy=False, nan=-1)
        area = self._calc_area_from_keypoints(keypoints)

        res = {}
        res["image"] = image
        res[
            "original_size"
        ] = original_size  # In order to convert back the keypoints to their original space
        res["annotations"] = {}
        res["annotations"]["keypoints"] = keypoints
        res["annotations"]["area"] = area
        res["annotations"]["ids"] = ids
        res["annotations"]["boxes"] = bbox_tensor
        res["annotations"]["image_id"] = image_id
        res["annotations"]["is_crowd"] = is_crowd
        res["annotations"]["labels"] = labels

        return res


class CroppedDataset(Dataset, BaseDataset):
    def __init__(
        self, project: DLCProject, transform: object = None, mode: str = "train"
    ):
        super().__init__()
        self.transform = transform
        self.project = project
        self.cfg = self.project.cfg
        self.bodyparts = auxiliaryfunctions.get_bodyparts(self.cfg)
        self.num_joints = len(self.bodyparts)
        self.shuffle = self.project.shuffle
        self.project.convert2dict(mode)
        self.dataframe = self.project.dataframe

        self.annotations = self._compute_anno()

        modelfolder = os.path.join(
            self.project.proj_root,
            get_model_folder(
                self.cfg["TrainingFraction"][0],
                self.shuffle,
                self.cfg,
                "",
            ),
        )
        pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
        pytorch_cfg = read_plainconfig(pytorch_config_path)
        self.with_center = pytorch_cfg.get("with_center", False)
        self.individuals = self.cfg.get("individuals", ["single"])
        self.individual_to_idx = {}
        for i, indiv in enumerate(self.individuals):
            self.individual_to_idx[indiv] = i
        self.max_num_animals = len(self.individuals)
        self.color_mode = pytorch_cfg.get("colormode", "RGB")

        self.input_size = 256, 256  # (h, w) #TODO make that depend on pytorch config
        self.crop = A.Compose(
            [
                A.RandomCropNearBBox(
                    max_part_shift=0.0,
                    cropping_box_key="animal_bbox",
                    always_apply=True,
                    p=1.0,
                ),
                A.Resize(*self.input_size),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            bbox_params=A.BboxParams(format="coco"),
        )

        # We must drop na because self.project.images doesn't contain imgaes with no
        # labels so it can produce an indexnotfound error
        # length is stored here to avoid repeating the computation
        self.length = len(self.annotations)

    def __len__(self):
        return self.length
    
    def _keypoint_in_boundary(self, keypoint: list, shape: tuple):
        """Summary:
        Check if a keypoint lies inside the given shape.

        Args:
            keypoint: [x,y] coordinates of the keypoints
            shape: tuple representing the shape of the boundary (height, width)

        Returns:
            Whether a keypoint lies inside the given shape

        Example:
            input:
                keypoint = [100, 50]
                shape = (200, 300)
            output:
                _keypoint_in_boundary(keypoint, shape) = True
        """
        return (
            (keypoint[0] > 0)
            and (keypoint[1] > 0)
            and (keypoint[0] < shape[1])
            and (keypoint[1] < shape[0])
        )

    def _compute_anno(self):
        """Summary:
        Compute annotations for the dataset

        Args:
            None

        Returns:
            annotations: list of annotations containing information about keypoints and image paths.
        """
        annotations = []
        df_length = self.dataframe.shape[0]

        for index in range(df_length):
            image_path = self.dataframe.index[index]
            if isinstance(image_path, tuple):
                image_path = os.path.join(self.cfg["project_path"], *image_path)
            else:
                image_path = os.path.join(self.cfg["project_path"], image_path)

            image_id = self.project.image_path2image_id[image_path]
            annotations_ids = self.project.id2annotations_idx[image_id]
            for ann_idx in annotations_ids:
                ann = self.project.annotations[ann_idx]
                if (
                    ann["bbox"] == 0.0
                ).all():  # I think we don't want unnanotated cropped images
                    continue
                ann["image_path"] = image_path
                annotations.append(ann)

        return annotations

    def _calc_area_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        w = keypoints[:, 0].max(axis=0) - keypoints[:, 0].min(axis=0)
        h = keypoints[:, 1].max(axis=0) - keypoints[:, 1].min(axis=0)
        return w * h

    def __getitem__(self, index: int) -> dict:
        # load images
        ann = self.annotations[index]
        image_file = ann["image_path"]

        image = cv2.imread(image_file)
        if self.color_mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape

        if not self.with_center:
            keypoints = np.zeros((self.num_joints, 3))
            num_keypoints_returned = self.num_joints
        else:
            keypoints = np.zeros((self.num_joints + 1, 3))
            num_keypoints_returned = self.num_joints + 1

        _keypoints, _undef_ids = self.project.annotation2keypoints(ann)
        if self.with_center:
            keypoints[:-1, :2] = np.array(_keypoints)
            keypoints[:-1, 2] = _undef_ids
        else:
            keypoints[:, :2] = np.array(_keypoints)
            keypoints[:, 2] = _undef_ids
        animal_id = self.individual_to_idx[ann["individual"]]

        crop_box = ann["bbox"].copy()
        crop_box[2] += crop_box[0]
        crop_box[3] += crop_box[1]
        cropped = self.crop(
            image=image, keypoints=keypoints[:, :2], bboxes=[], animal_bbox=crop_box
        )
        image = cropped["image"]
        keypoints = [
            (-1, -1) if (keypoints[i, 2] == 0) else keypoint
            for i, keypoint in enumerate(cropped["keypoints"])
        ]

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            shape_transformed = transformed["image"].shape
            transformed["keypoints"] = [
                (-1, -1)
                if not self._keypoint_in_boundary(keypoint, shape_transformed)
                else keypoint
                for i, keypoint in enumerate(transformed["keypoints"])
            ]
        else:
            transformed = {}
            transformed["keypoints"] = keypoints
            transformed["image"] = image

        image = torch.tensor(transformed["image"], dtype=torch.float).permute(
            2, 0, 1
        )  # channels first

        assert len(transformed["keypoints"]) == len(keypoints)
        keypoints = np.array(transformed["keypoints"]).astype(float)

        # Center keypoint needs to be computed after transformation because
        # it should depend on visible keypoints only (which may change after augmentation)
        if self.with_center:
            try:
                keypoints[-1, :] = np.nanmean(
                    keypoints[:-1, :][~np.any(keypoints[:-1, :] == -1, axis=1)].reshape(
                        -1, 2
                    ),
                    axis=0,
                )
            except ValueError:
                keypoints[-1, :] = np.array([-1, -1])
        np.nan_to_num(keypoints, copy=False, nan=-1)
        area = self._calc_area_from_keypoints(keypoints)

        # Animal_idx is always the first dimension even is there is only one animal
        # This convention is the one adopted int this whole repository
        res = {}
        res["image"] = image
        res[
            "original_size"
        ] = original_size  # In order to convert back the keypoints to their original space
        res["annotations"] = {}
        res["annotations"]["keypoints"] = keypoints[None, :]
        res["annotations"]["area"] = np.array(area)[None]
        res["annotations"]["ids"] = np.array(animal_id)[None]
        res["annotations"]["path"] = image_file

        return res
