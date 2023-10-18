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

from deeplabcut.pose_estimation_pytorch.data.base import Loader


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
    train_json_filename: str = "train.json"
    test_json_filename: str | None = "test.json"

    def __post_init__(self) -> None:
        self.train_json = self._load_json(self.project_root, self.train_json_filename)
        self.test_json = None
        if self.test_json_filename:
            self.test_json = self._load_json(self.project_root, self.test_json_filename)

        # TODO: change when _get_dataset_parameters is abstract
        self.cfg = {}

    @staticmethod
    def _load_json(project_root: str, filename: str) -> dict:
        """Load a JSON file from the annotations directory.

        Args:
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
            json_obj = self.train_json
        elif mode == "test":
            json_obj = self.test_json
        else:
            raise AttributeError(f"Unknown mode: {mode}")

        for image in json_obj["images"]:
            image_path = image["file_name"]
            image["file_name"] = os.path.join(
                self.project_root,
                "images",
                image_path,
            )

        return json_obj
