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

import json
import os
from dataclasses import dataclass

import numpy as np

from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.utils import (
    map_id_to_annotations,
    map_image_path_to_id,
)


@dataclass
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
            train_json_filename="train.json",
            test_json_filename="test.json",
        )
    """

    project_root: str
    model_config_path: str
    train_json_filename: str = "train.json"
    test_json_filename: str | None = "test.json"

    def __post_init__(self) -> None:
        super().__init__(self.project_root, self.model_config_path)
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
        num_individuals, bodyparts = self.get_project_parameters(self.train_json)
        return PoseDatasetParameters(
            bodyparts=bodyparts,
            unique_bpts=[],
            individuals=[f"individual{i}" for i in range(num_individuals)],
            with_center_keypoints=self.model_cfg.get("with_center_keypoints", False),
            color_mode=self.model_cfg.get("color_mode", "RGB"),
            cropped_image_size=self.model_cfg.get("output_size", (256, 256)),
        )

    @staticmethod
    def load_json(project_root: str, filename: str) -> dict:
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

    def load_data(self, mode: str = "train") -> dict:
        """Convert data from JSON object to dictionary.
        Args:
            mode: indicates which JSON object to convert. Defaults to "train".

        Returns:
            the train or test data
        """
        # todo: add validation
        if mode == "train":
            data = self.train_json
        elif mode == "test":
            data = self.test_json
        else:
            raise AttributeError(f"Unknown mode: {mode}")

        for image in data["images"]:
            image_path = image["file_name"]
            image["file_name"] = os.path.join(self.project_root, "images", image_path)

        for annotation in data["annotations"]:
            annotation["keypoints"] = np.array(annotation["keypoints"], dtype=float)
            annotation["bbox"] = np.array(annotation["bbox"], dtype=float)
            annotation["individual"] = "unknown"

        annotations_with_bbox = self._get_all_bboxes(
            data["images"],
            data["annotations"],
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
        num_individuals = max(*[len(a_ids) for a_ids in img_to_annotations.values()])
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
