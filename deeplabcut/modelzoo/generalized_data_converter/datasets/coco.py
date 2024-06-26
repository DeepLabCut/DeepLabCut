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
import copy
import json
import os

from deeplabcut.modelzoo.generalized_data_converter.datasets.base import BasePoseDataset


class COCOPoseDataset(BasePoseDataset):
    def __init__(
        self,
        proj_root,
        dataset_name,
        train_filename="train.json",
        shuffle=None,
    ):

        super(COCOPoseDataset, self).__init__()

        self.meta["dataset_name"] = dataset_name
        self.meta["proj_root"] = proj_root

        self.proj_root = proj_root
        self.annotations_by_category = {}

        self.train_json_obj = (
            self._load_json(train_filename)
            if shuffle is None
            else self._load_json(
                train_filename.replace(".json", f"_shuffle{shuffle}.json")
            )
        )
        self.test_json_obj = (
            self._load_json("test.json")
            if shuffle is None
            else self._load_json(f"test_shuffle{shuffle}.json")
        )

        self.populate_generic()

    def _load_json(self, json_fn):
        path = os.path.join(self.proj_root, "annotations", json_fn)
        with open(path, "r") as f:
            json_obj = json.load(f)
        return json_obj

    def populate_generic(self):

        temp_train_images = copy.deepcopy(self.train_json_obj["images"])
        temp_test_images = copy.deepcopy(self.test_json_obj["images"])

        for image in temp_train_images + temp_test_images:
            image_path = image["file_name"]
            # if os.sep not in image_path:
            # assuming the file_name is mmpose style, i.e. only the image name is stored
            # so we need to add back absolute path

            image["file_name"] = os.path.join(self.proj_root, "images", image_path)

        self.generic_train_images = temp_train_images
        self.generic_test_images = temp_test_images

        self.generic_train_annotations = self.train_json_obj["annotations"]

        self.generic_test_annotations = self.test_json_obj["annotations"]

        self.meta["categories"] = self.test_json_obj["categories"][0]

        self._build_maps()

        print(f"Before checking trainset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_train_images, self.generic_train_annotations
        )

        print(f"Before checking testset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_test_images, self.generic_test_annotations
        )
