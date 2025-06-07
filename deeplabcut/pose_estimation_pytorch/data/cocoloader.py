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

import json
import os
import warnings
from pathlib import Path

import numpy as np

from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.utils import (
    map_id_to_annotations,
    map_image_path_to_id,
)


class COCOLoader(Loader):
    """
    Attributes:
        project_root: root directory path of the COCO project.
        train_json_filename: the name of the json file containing the train annotations
        test_json_filename: the name of the json file containing the train annotations.
            None if there is no test set.

    Examples:
        loader = COCOLoader(
            project_root='/path/to/project/',
            model_config_path='/path/to/project/experiments/train/pytorch_config.yaml'
            train_json_filename="train.json",
            test_json_filename="test.json",
        )
    """

    def __init__(
        self,
        project_root: str | Path,
        model_config_path: str | Path,
        train_json_filename: str = "train.json",
        test_json_filename: str = "test.json",
    ):
        image_root = Path(project_root) / "images"
        super().__init__(project_root, image_root, Path(model_config_path))
        self.train_json_filename = train_json_filename
        self.test_json_filename = test_json_filename
        self._dataset_parameters = None

        self.train_json = self.load_json(self.project_root, self.train_json_filename)
        self.test_json = None
        if self.test_json_filename:
            self.test_json = self.load_json(self.project_root, self.test_json_filename)

    def get_dataset_parameters(self) -> PoseDatasetParameters:
        """
        Retrieves dataset parameters based on the instance's configuration.

        Returns:
            An instance of the PoseDatasetParameters with the parameters set.
        """
        if self._dataset_parameters is None:
            num_individuals, bodyparts = self.get_project_parameters(self.train_json)

            crop_cfg = self.model_cfg["data"]["train"].get("top_down_crop", {})
            crop_w, crop_h = crop_cfg.get("width", 256), crop_cfg.get("height", 256)
            crop_margin = crop_cfg.get("margin", 0)
            crop_with_context = crop_cfg.get("crop_with_context", True)

            self._dataset_parameters = PoseDatasetParameters(
                bodyparts=bodyparts,
                unique_bpts=[],
                individuals=[f"individual{i}" for i in range(num_individuals)],
                with_center_keypoints=self.model_cfg.get("with_center_keypoints", False),
                color_mode=self.model_cfg.get("color_mode", "RGB"),
                top_down_crop_size=(crop_w, crop_h),
                top_down_crop_margin=crop_margin,
                top_down_crop_with_context=crop_with_context,
            )

        return self._dataset_parameters

    @staticmethod
    def load_json(project_root: str | Path, filename: str) -> dict:
        """Load a JSON file from the annotations directory.

        Args:
            project_root: path to the root directory for the project
            filename: filename of JSON file to load

        Returns:
            json_obj: JSON object loaded from the file

        Raises:
            FileNotFoundError if the file does not exist
            ValueError if the object stored in the file is not a dict

        Examples:
            Check https://docs.trainingdata.io/v1.0/Export%20Format/COCO/ to see
            examples of how a json file looks like.
        """
        json_path = os.path.join(project_root, "annotations", filename)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File {json_path} does not exist.")

        with open(json_path, "r") as f:
            json_obj = json.load(f)

        if not isinstance(json_obj, dict):
            raise ValueError("COCO datasets need to be saved in JSON Objects")

        return json_obj

    @staticmethod
    def validate_categories(coco_json: dict) -> dict:
        """Checks that the categories for the COCO project are valid.

        Checks that there is no category with ID 0 in the dataset, as this causes issues
        with torchvision object detectors (label 0 is reserved for background
        detections). If that's the case, all category IDs are shifted by 1 such that
        there is no longer a category 0.

        Currently, detectors can only be trained with a single category. This also
        ensures that all annotations have `category_id` set to 1.

        Args:
            coco_json: the COCO dictionary containing the annotations

        Returns:
            the validated COCO object
        """
        cat_0 = False
        for cat in coco_json["categories"]:
            if cat["id"] == 0:
                cat_0 = cat
                warnings.warn(
                    f"Found a category with ID 0 ({cat}) in the COCO dataset. This is not"
                    f" allowed, as category ID 0 is reserved as the background ID for"
                    f" torchvision detectors. All category IDs have been shifted by 1."
                )

        if len(coco_json["categories"]) > 1:
            warnings.warn(
                f"Found more than 1 category in the project. This is currently not"
                f" supported in DeepLabCut. All annotations will be given category 1"
            )

        if cat_0:
            for cat in coco_json["categories"]:
                cat["id"] = 1

        if cat_0 or len(coco_json["categories"]) > 1:
            for ann in coco_json["annotations"]:
                ann["category_id"] = 1

        return coco_json

    def validate_images(self, coco_json: dict) -> dict:
        """Goes over images and annotations to look for potential errors

        This code tries to ensure that training a model on this project does not crash
        down the line

        Completes relative image filepaths to '/project_root/images/file_name'. Absolute
        filepaths are not updated (which allows storing images to be stored in a folder
        other than the project root) Then checks that all images files exist in the file
        system.

        Args:
            project_root: the root path of the COCO project
            coco_json: the COCO dictionary containing the annotations

        Returns:
            the validated COCO object
        """
        image_ids = set()
        missing_images = {}
        validated_images = []
        for image in coco_json["images"]:
            image_filename = Path(image["file_name"])
            if image_filename.is_absolute():
                image_path = image_filename
            else:
                image_path = self.image_root / image["file_name"]
                image["file_name"] = str(image_path)

            if not image_path.exists():
                missing_images[image["id"]] = image["file_name"]
            else:
                validated_images.append(image)
                image_ids.add(image["id"])

        if len(missing_images) > 0:
            warnings.warn(
                f"There are {len(missing_images)} images that cannot be found (here"
                " are some):"
            )
            for img_id, file_name in missing_images.items():
                print(f"  * {img_id}: {file_name}")

        coco_json["images"] = validated_images

        if len(missing_images) > 0:
            validated_annotations = []
            for ann in coco_json["annotations"]:
                if ann["image_id"] not in missing_images:
                    validated_annotations.append(ann)

            coco_json["annotations"] = validated_annotations

        validated_annotations = []
        for ann in coco_json["annotations"]:
            if ann["image_id"] in image_ids:
                validated_annotations.append(ann)

        if len(coco_json["annotations"]) < len(validated_annotations):
            warnings.warn(
                f"Found some annotations for which the image ID was not in the images."
                f" Removing them from the dataset."
            )
            print(f"  All annotations: {len(coco_json['annotations'])}")
            print(f"  Annotations with correct image IDs: {len(validated_annotations)}")
            coco_json["annotations"] = validated_annotations

        return coco_json

    def load_data(self, mode: str = "train") -> dict:
        """Convert data from JSON object to dictionary.
        Args:
            mode: indicates which JSON object to convert. Defaults to "train".

        Returns:
            the train or test data
        """
        if mode == "train":
            data = self.train_json
        elif mode == "test":
            data = self.test_json
        else:
            raise AttributeError(f"Unknown mode: {mode}")

        data = COCOLoader.validate_categories(data)
        data = self.validate_images(data)

        annotations_per_image = {}
        for annotation in data["annotations"]:
            annotation["keypoints"] = np.array(annotation["keypoints"], dtype=float)
            annotation["bbox"] = np.array(annotation["bbox"], dtype=float)

            # set individual index
            image_id = annotation["image_id"]
            individual_idx = annotations_per_image.get(image_id, 0)
            annotation["individual"] = f"individual{individual_idx}"
            annotations_per_image[image_id] = individual_idx + 1

        filter_annotations = []
        for annotation in data['annotations']:
            keypoints = annotation['keypoints']
            bbox = annotation['bbox']
            if np.all(keypoints <= 0) or len(bbox) == 0:
                continue
            filter_annotations.append(annotation)

        data["annotations"] = filter_annotations        
        
        # FIXME: why estimating bbox when there are already bbox?
        annotations_with_bbox = self._compute_bboxes(
            data["images"],
            data["annotations"],
            method="gt",
        )
        data["annotations"] = annotations_with_bbox
        return data

    @staticmethod
    def get_project_parameters(train_json: dict) -> tuple[int, list[str]]:
        """
        Loads the parameters for the project from the train json file
        TODO: Should this compute the number also using the test json?

        Args:
            train_json: the json dictionary containing the data for training

        Returns:
            int: the maximum number of individuals in a single image
            list[str]: the name of keypoints annotated in this project
        """
        # TODO: Check that there's a single category
        bodyparts = train_json["categories"][0]["keypoints"]

        img_to_annotations = map_id_to_annotations(train_json["annotations"])
        if len(img_to_annotations) == 0:
            raise ValueError(f"No images found in the dataset: {train_json}!")
        elif len(img_to_annotations) == 1:
            num_individuals = len(list(img_to_annotations.values())[0])
        else:
            num_individuals = max(
                *[len(a_ids) for a_ids in img_to_annotations.values()]
            )

        return num_individuals, bodyparts

    def predictions_to_coco(
        self,
        predictions: dict[str, dict[str, np.ndarray]],
        mode: str = "train",
    ) -> list[dict]:
        """Converts detections to COCO format

        Args:
            predictions: a dictionary mapping image name to the predictions made for it
            mode: {"train", "test"} the mode that the predictions were made with

        Returns:
            The COCO-format predictions
        """
        data = self.load_data(mode)
        image_path_to_id = map_image_path_to_id(data["images"])

        # TODO: no unique bodyparts for COCO
        coco_predictions = []
        for image_path, pred in predictions.items():
            image_id = image_path_to_id[image_path]

            # Shape (num_individuals, num_keypoints, 3)
            individuals = pred["bodyparts"]
            for idx, keypoints in enumerate(individuals):
                if not np.all(keypoints == -1):
                    score = np.mean(keypoints[:, 2]).item()
                    keypoints = keypoints.copy()
                    keypoints[:, 2] = 2  # set visibility instead of score
                    coco_pred = {
                        "image_id": int(image_id),
                        "category_id": 1,  # TODO: get category ID from prediction?
                        "keypoints": keypoints.reshape(-1).tolist(),
                        "score": float(score),
                    }
                    if "bboxes" in pred:
                        coco_pred["bbox"] = pred["bboxes"][idx].reshape(-1).tolist()
                    if "bbox_scores" in pred:
                        coco_pred["bbox_scores"] = (
                            pred["bbox_scores"][idx].reshape(-1).tolist()
                        )

                    coco_predictions.append(coco_pred)

        return coco_predictions
