# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import json
import os
from typing import List

from .base import BaseProject


class COCOProject(BaseProject):
    """
    Definition of the class object COCOProject.
    """

    def __init__(
        self,
        proj_root: str,
        shuffle: int = 0,
        image_id_offset: int = 0,
        keys_to_load: List[str] = ["images", "annotations"],
    ):
        """Summary:
        Constructor of the COCOProject.
        Loads the data

        Args:
            proj_root: root directory path of the COCO project.
            shuffle: shuffle value used to select a specific shuffle of train and test JSON files. Defaults to 0.
            image_id_offset: the offset value to be added to image IDs. Defaults to 0.
            keys_to_load: list of strings specifying the keys to load from the JSON files. Defaults to ["images", "annotations"].

        Returns:
            None

        Examples:
            project = COCOProject( proj_root = 'path/to/project/', shuffle = 1, image_id_offset = 1000, keys_to_load = ["images", "annotations"] )
        """
        super().__init__()
        self.proj_root = proj_root
        self.keys_to_load = keys_to_load

        self.train_json_obj = (
            self._load_json("train.json")
            if shuffle is None
            else self._load_json(f"train_shuffle{shuffle}.json")
        )
        self.test_json_obj = (
            self._load_json("test.json")
            if shuffle is None
            else self._load_json(f"test_shuffle{shuffle}.json")
        )

    def _load_json(self, json_fn):
        """Summary:
        Load a JSON file from the annotations directory.

        Args:
            json_fn: filename of JSON file to load

        Returns:
            json_obj: JSON object loaded from the file

        Examples:
            Check https://docs.trainingdata.io/v1.0/Export%20Format/COCO/ to see
            examples of how a json file looks like.
        """
        path = os.path.join(self.proj_root, "annotations", json_fn)
        with open(path, "r") as f:
            json_obj = json.load(f)
        return json_obj

    def load_split(self):
        """Summary:
        We expected that coco project has train test split in train test json already

        Args:
            None

        Return:
            None
        """
        pass

    def convert2dict(self, mode: str = "train"):
        """Summary:
        Convert data from JSON objecy to dictionary.

        Args:
            mode: indicates which JSON object to convert. Defaults to "train".

        Returns:
            None

        Examples:
            mode = 'test'
        """
        json_obj = getattr(self, f"{mode}_json_obj")

        for image in self.images:
            image_path = image["file_name"]
            # if os.sep not in image_path:
            # assuming the file_name is mmpose style, i.e. only the image name is stored
            # so we need to add back absolute path
            image["file_name"] = os.path.join(self.proj_root, "images", image_path)

        for key in self.keys_to_load:
            setattr(self, key, json_obj[key])
