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

from abc import ABC, abstractmethod
from pathlib import Path

import albumentations as A
import numpy as np

import deeplabcut.pose_estimation_pytorch.config as config
from deeplabcut.pose_estimation_pytorch.data.dataset import (
    PoseDataset,
    PoseDatasetParameters,
)
from deeplabcut.pose_estimation_pytorch.data.utils import (
    _compute_crop_bounds,
    bbox_from_keypoints,
    map_id_to_annotations,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions


class Loader(ABC):
    """
    Abstract class that represents a blueprint for loading and processing dataset information.

    Methods:
        load_data(mode: str = 'train') -> dict:
            Abstract method to convert the project configuration to a standard COCO format.
        create_dataset(images: dict = None, annotations: dict = None, transform: object = None,
            mode: str = "train", task: Task = Task.BOTTOM_UP) -> PoseDataset:
            Creates and returns a PoseDataset given a set of images, annotations, and other parameters.
        _compute_bboxes(images, annotations, method: str = 'gt') -> dict:
            Retrieves all bounding boxes based on the specified method.
        get_dataset_parameters(*args, **kwargs) -> dict:
            Returns a dictionary containing dataset parameters derived from the configuration.
    """

    def __init__(self, model_config_path: str | Path) -> None:
        self.model_config_path = Path(model_config_path)
        self.model_cfg = config.read_config_as_dict(str(model_config_path))
        self._loaded_data: dict[str, dict[str, list[dict]]] = {}

    @property
    def model_folder(self) -> Path:
        """Returns: The path of the folder containing the model data"""
        return self.model_config_path.parent

    def update_model_cfg(self, updates: dict) -> None:
        """Updates the model configuration

        Args:
            updates: the items to update in the model configuration
        """
        self.model_cfg = config.update_config(self.model_cfg, updates)
        config.write_config(self.model_config_path, self.model_cfg)

    @abstractmethod
    def load_data(self, mode: str = "train") -> dict[str, list[dict]]:
        """Abstract method to convert the project configuration to a standard coco format.

        Raises:
            NotImplementedError: This method must be implemented in the derived classes.
        """
        raise NotImplementedError

    def image_filenames(self, mode: str = "train") -> list[str]:
        """
        Args:
            mode: {"train", "test"} whether to load train or test data

        Returns:
            the image paths for this mode
        """
        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)

        data = self._loaded_data[mode]
        return [image["file_name"] for image in data["images"]]

    def ground_truth_keypoints(
        self, mode: str = "train", unique_bodypart: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Creates a dictionary containing the ground truth data

        TODO: make more efficient

        Args:
            mode: {"train", "test"} whether to load train or test data
            unique_bodypart: returns the ground truth for unique bodyparts

        Raises:
            ValueError if unique_bodypart=True but there are no unique bodyparts

        Returns:
            A dict mapping image paths to the ground truth annotations for the mode in
            the format:
                {'image': keypoints with shape (num_individuals, num_keypoints, 2)}
        """
        parameters = self.get_dataset_parameters()
        if unique_bodypart:
            if not parameters.num_unique_bpts > 0:
                raise ValueError("There are no unique bodyparts in this dataset!")
            individuals = ["single"]
            num_bodyparts = parameters.num_unique_bpts
        else:
            individuals = parameters.individuals
            num_bodyparts = parameters.num_joints

        if "weight_init" in self.model_cfg["train_settings"]:
            weight_init_cfg = self.model_cfg["train_settings"]["weight_init"]
            if weight_init_cfg["memory_replay"]:
                conversion_array = weight_init_cfg["conversion_array"]
                num_bodyparts = len(conversion_array)

        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)
        data = self._loaded_data[mode]

        annotations = self.filter_annotations(data["annotations"], task=Task.BOTTOM_UP)
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            individual_keypoints = {
                annotations[i]["individual"]: annotations[i]["keypoints"]
                for i in img_to_ann_map[image["id"]]
            }
            gt_array = np.zeros((len(individuals), num_bodyparts, 3))
            # Keep the shape of the ground truth
            for idv_idx, idv in enumerate(individuals):
                if idv in individual_keypoints:
                    keypoints = individual_keypoints[idv].reshape(num_bodyparts, -1)
                    gt_array[idv_idx, :, :] = keypoints[:, :3]

            ground_truth_dict[image_path] = gt_array

        return ground_truth_dict

    def ground_truth_bboxes(self, mode: str = "train") -> dict[str, np.ndarray]:
        """Creates a dictionary containing the ground truth bounding boxes

        Args:
            mode: {"train", "test"} whether to load train or test data

        Returns:
            A dict mapping image paths to the ground truth annotations for the mode in
            the format:
                {'image': bboxes with shape (num_individuals, xywh)}
        """
        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)
        data = self._loaded_data[mode]

        annotations = self.filter_annotations(data["annotations"], task=Task.DETECT)
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            img_shape = image["height"], image["width"], 3
            bboxes = [annotations[i]["bbox"] for i in img_to_ann_map[image["id"]]]
            if len(bboxes) == 0:
                bboxes = np.zeros((0, 4))
            else:
                bboxes = _compute_crop_bounds(np.stack(bboxes, axis=0), img_shape)
            ground_truth_dict[image_path] = bboxes

        return ground_truth_dict

    def create_dataset(
        self,
        transform: A.BaseCompose | None = None,
        mode: str = "train",
        task: Task = Task.BOTTOM_UP,
    ) -> PoseDataset:
        """
        Creates a PoseDataset based on provided arguments.

        Args:
            transform: Transformation to be applied on dataset. Defaults to None.
            mode: Mode in which dataset is to be used (e.g., 'train', 'test'). Defaults to 'train'.
            task: Task for which the dataset is being used. Defaults to 'BU'.

        Returns:
            PoseDataset: An instance of the PoseDataset class.

        Raises:
            Any exception raised by `get_dataset_parameters` or `load_data` methods.
        """
        parameters = self.get_dataset_parameters()
        data = self.load_data(mode)
        data["annotations"] = self.filter_annotations(data["annotations"], task)
        dataset = PoseDataset(
            images=data["images"],
            annotations=data["annotations"],
            transform=transform,
            mode=mode,
            task=task,
            parameters=parameters,
        )
        return dataset

    @abstractmethod
    def get_dataset_parameters(self) -> PoseDatasetParameters:
        """
        Retrieves dataset parameters based on the instance's configuration.

        Returns:
            An instance of the PoseDatasetParameters with the parameters set.
        """
        raise NotImplementedError

    @staticmethod
    def filter_annotations(annotations: list[dict], task: Task) -> list[dict]:
        """Filters annotations based on the task, removing empty annotations

        For pose estimation tasks, annotations with empty keypoints are removed. For
        detection task, annotations with no bounding boxes are removed

        Args:
            annotations: the annotations to filter
            task: the task for which to filter

        Returns:
            list: the filtered annotations
        """
        filtered_annotations = []
        for annotation in annotations:
            keypoints = annotation["keypoints"].reshape(-1, 3)
            if task in (Task.DETECT, Task.TOP_DOWN) and (
                annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0
            ):
                continue
            elif task != Task.DETECT and np.all(keypoints[:, :2] <= 0):
                continue

            filtered_annotations.append(annotation)

        return filtered_annotations

    @staticmethod
    def _compute_bboxes(
        images: list[dict],
        annotations: list[dict],
        method: str = "gt",
    ):
        """TODO: Nastya method of bbox computation (detection bbox, seg. mask, ...)
        Retrieves all bounding boxes based on the given method.

        Args:
            images: A list of images.
            annotations: A list of annotations corresponding to images.
            method (str, optional): Method to use for retrieving bounding boxes. Defaults to 'gt'.
                - 'gt': Ground truth bounding boxes.
                - 'detection bbox': Bounding boxes from detection.
                - 'keypoints': Bounding boxes from keypoints.
                - 'segmentation mask': Bounding boxes from segmentation masks.

        Returns:
            list: Updated annotations based on the given method.

        Raises:
            ValueError: If 'bbox' is not found in annotation when method is 'gt'.
            ValueError: If method is not one of 'gt', 'detection bbox', 'keypoints', or 'segmentation mask'.
        """

        if not method:
            return annotations

        elif method == "gt":
            for i, annotation in enumerate(annotations):
                if "bbox" not in annotation:
                    # or do something else?
                    raise ValueError(
                        f"Bounding box not found in annotation {annotation}, please "
                        "chose another bbox computation method"
                    )
            return annotations

        elif method == "detection bbox":
            raise NotImplementedError

        elif method == "keypoints":
            bbox_margin = 20  # TODO: should not be hardcoded
            min_area = 1  # TODO: should not be hardcoded
            img_id_to_annotations = map_id_to_annotations(annotations)
            for img in images:
                anns = [annotations[idx] for idx in img_id_to_annotations[img["id"]]]
                for a in anns:
                    a["bbox"] = bbox_from_keypoints(
                        keypoints=a["keypoints"],
                        image_h=img["height"],
                        image_w=img["width"],
                        margin=bbox_margin,
                    )
                    a["area"] = max(min_area, (a["bbox"][2] * a["bbox"][3]).item())
            return annotations

        elif method == "segmentation mask":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown method: {method}")
