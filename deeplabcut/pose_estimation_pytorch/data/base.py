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

from abc import ABC, abstractmethod

import albumentations as A
import numpy as np

from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.data.dataset import (
    PoseDataset,
    PoseDatasetParameters,
)
from deeplabcut.pose_estimation_pytorch.data.utils import (
    _compute_crop_bounds,
    map_id_to_annotations,
)
from deeplabcut.utils.auxiliaryfunctions import get_bodyparts, get_unique_bodyparts


class Loader(ABC):
    """
    Abstract class that represents a blueprint for loading and processing dataset information.

    Methods:
        load_data(mode: str = 'train') -> dict:
            Abstract method to convert the project configuration to a standard COCO format.
        create_dataset(images: dict = None, annotations: dict = None, transform: object = None, mode: str = "train", task: str = "BU") -> PoseDataset:
            Creates and returns a PoseDataset given a set of images, annotations, and other parameters.
        _get_all_bboxes(images, annotations, method: str = 'gt') -> dict:
            Retrieves all bounding boxes based on the specified method.
        _get_dataset_parameters(*args, **kwargs) -> dict:
            Returns a dictionary containing dataset parameters derived from the configuration.
    """

    def __init__(self, project_root: str, model_config_path: str) -> None:
        self.project_root = project_root
        self.model_config_path = model_config_path
        self.model_cfg = auxiliaryfunctions.read_plainconfig(model_config_path)
        self._loaded_data: dict[str, dict[str, dict]] = {}
        self._get_dataset_parameters()

    @abstractmethod
    def load_data(self, mode: str = "train") -> dict[str, dict]:
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
        parameters = self._get_dataset_parameters()
        if unique_bodypart:
            if not parameters.num_unique_bpts > 0:
                raise ValueError("There are no unique bodyparts in this dataset!")
            individuals = ["single"]
            num_bodyparts = parameters.num_unique_bpts
        else:
            individuals = parameters.individuals
            num_bodyparts = parameters.num_joints

        if mode not in self._loaded_data:
            self._loaded_data[mode] = self.load_data(mode)
        data = self._loaded_data[mode]

        annotations = self.filter_annotations(data["annotations"])
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            individual_keypoints = {
                annotations[i]["individual"]: annotations[i]["keypoints"]
                for i in img_to_ann_map[image["id"]]
            }
            gt_array = np.empty((len(individuals), num_bodyparts, 3))
            gt_array.fill(np.nan)

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

        annotations = self.filter_annotations(data["annotations"])
        img_to_ann_map = map_id_to_annotations(annotations)

        ground_truth_dict = {}
        for image in data["images"]:
            image_path = image["file_name"]
            img_shape = image["height"], image["width"], 3
            bboxes = [annotations[i]["bbox"] for i in img_to_ann_map[image["id"]]]
            ground_truth_dict[image_path] = _compute_crop_bounds(
                np.stack(bboxes, axis=0), img_shape
            )

        return ground_truth_dict

    def create_dataset(
        self,
        transform: A.BaseCompose | None = None,
        mode: str = "train",
        task: str = "BU",
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
            Any exception raised by `_get_dataset_parameters` or `load_data` methods.
        """
        parameters = self._get_dataset_parameters()
        data = self.load_data(mode)
        data["annotations"] = self.filter_annotations(data["annotations"])
        dataset = PoseDataset(
            images=data["images"],
            annotations=data["annotations"],
            transform=transform,
            mode=mode,
            task=task,
            parameters=parameters,
        )
        return dataset

    def _get_dataset_parameters(self, *args, **kwargs) -> PoseDatasetParameters:
        """TODO: _get_dataset_parameters should be an abstract method
        Retrieves dataset parameters based on the instance's configuration.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An instance of the PoseDatasetParameters with the parameters set.
        """
        return PoseDatasetParameters(
            bodyparts=get_bodyparts(self.cfg),
            unique_bpts=get_unique_bodyparts(self.cfg),
            individuals=self.cfg.get("individuals", ["animal"]),
            with_center_keypoints=self.model_cfg.get("with_center_keypoints", False),
            color_mode=self.model_cfg.get("color_mode", "RGB"),
            cropped_image_size=self.model_cfg.get("output_size", (256, 256)),
        )

    @staticmethod
    def filter_annotations(annotations: list[dict]) -> list[dict]:
        """Filters annotations based on the keypoints, removing empty annotations

        Args:
            annotations: A list of annotations.

        Returns:
            list: A list of filtered annotations.
        """
        filtered_annotations = []
        for annotation in annotations:
            keypoints = annotation["keypoints"].reshape(-1, 3)
            annotation["bbox"].reshape(-1, 4)
            if np.all(keypoints[:, :2] <= 0):
                continue
            filtered_annotations.append(annotation)

        return filtered_annotations

    @staticmethod
    def _get_all_bboxes(images, annotations, method: str = "gt"):
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
                        "Bounding box not found in annotation, please chose another method"
                    )
            return annotations

        elif method == "detection bbox":
            return annotations

        elif method == "keypoints":
            return annotations

        elif method == "segmentation mask":
            return annotations

        else:
            raise ValueError(f"Unknown method: {method}")
