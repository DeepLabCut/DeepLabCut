import os
import pickle
from typing import List, Tuple

import deeplabcut
import numpy as np
import pandas as pd

from deeplabcut.pose_estimation_pytorch.utils import (
    df2generic
)
from deeplabcut.pose_estimation_pytorch.data.base import (
    BaseProject
)

class DLCProject(BaseProject):
    """
    Wrapper around the project containing information about the data,
    the actual annotations and the configs
    """

    def __init__(
        self,
        proj_root: str,
        shuffle: int = 0,
        image_id_offset: int = 0,
        keys_to_load: List[str] = ["images", "annotations"],
    ):
        """Summary:
        Constructor of the DLCProject class.
        Loads the data

        Args:
            proj_root: project path
            shuffle: shuffle index for the project. Defaults to 0.
            image_id_offset: offset value for image ids. Defaults to 0.
            keys_to_load: list of keys to load from the dataset.
                          Defaults to ["images", "annotations"].

        Return:
            None
        """

        super().__init__()
        self.proj_root = proj_root
        self.shuffle = shuffle
        self.keys_to_load = keys_to_load
        self.image_id_offset = image_id_offset
        config_file = os.path.join(self.proj_root, "config.yaml")
        self.cfg = deeplabcut.auxiliaryfunctions.read_config(config_file)
        self.task = self.cfg["Task"]
        self.scorer = self.cfg["scorer"]
        self.datasets_folder = os.path.join(
            self.proj_root,
            deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(self.cfg),
        )
        tr_frac = int(self.cfg["TrainingFraction"][0] * 100)
        self.path_dlc_data = os.path.join(
            self.datasets_folder, f"CollectedData_{self.scorer}.h5"
        )
        self.path_dlc_doc = os.path.join(
            self.datasets_folder,
            f"Documentation_data-{self.task}_{tr_frac}shuffle{self.shuffle}.pickle",
        )
        self.dlc_df = pd.read_hdf(self.path_dlc_data)
        self.load_split()
        self.dlc_df = self.dlc_df[~self.dlc_df.index.duplicated(keep="first")]
        self.df_train = self.df_train[~self.df_train.index.duplicated(keep="first")]
        if hasattr(self, "df_test"):
            self.df_test = self.df_test[~self.df_test.index.duplicated(keep="first")]

    def convert2dict(self, mode: str = "train"):
        """Summary:
        Convert the annotations dataframe into coco format dictionary of annotations

        Args:
            mode: mode indicating whether to use 'train' or 'test' data. Defaults to "train".

        Raises:
            AttributeError: if the specified mode (train or test) does not exist.

        Returns:
            None
        """
        try:
            self.dataframe = getattr(self, f"df_{mode}")
        except:
            raise AttributeError(
                f"PoseDataset doesn't have df_{mode} attr. Do project.train_test_split() first!"
            )

        data = df2generic(self.proj_root, self.dataframe, self.image_id_offset)

        self._init_annotation_image_correspondance(data)

        for key in self.keys_to_load:
            setattr(self, key, data[key])
        print("The data has been converted!")

    def _init_annotation_image_correspondance(self, data: dict):
        """Summary:
        Binds the image paths to corresponding annotations and ensures there is no indexing
        offsets between images and annotations when going through the dataset

        Args:
            data: dictionary containing annotations in COCO-like format

        Returns:
            None

        Examples:
            data = {"images": [...], "annotations": [...]}
        """
        # Path to id correspondence
        self.image_path2image_id = {}
        for i, image in enumerate(data["images"]):
            image_path = image["file_name"]
            self.image_path2image_id[image_path] = image["id"]

        # id to annotations list
        self.id2annotations_idx = {}
        for i, annotation in enumerate(data["annotations"]):
            image_id = annotation["image_id"]
            try:
                self.id2annotations_idx[image_id].append(i)
            except KeyError:
                self.id2annotations_idx[image_id] = [i]

        return

    def load_split(self):
        """Summary:
        Split the annotation dataframe into train and test dataframes based on project's split

        Args:
            None

        Return:
            None
        """
        with open(self.path_dlc_doc, "rb") as f:
            meta = pickle.load(f)

        train_ids = meta[1]
        test_ids = meta[2]

        train_images = self.dlc_df.index[train_ids]
        if len(test_ids) != 0:
            test_images = self.dlc_df.index[test_ids]
            self.dlc_images = np.hstack([train_images, test_images])
            self.df_test = self.dlc_df.loc[test_images]
        self.df_train = self.dlc_df.loc[train_images]

    @staticmethod
    def annotation2keypoints(annotation: dict) -> Tuple[list, np.array]:
        """Summary:
        Convert the coco annotations into array of keypoints also returns the array of the keypoints' visibility
        Args:
            annotation: dictionary containing coco-like annotations

        Returns:
            keypoints: paired keypoints
            undef_ids: array where 0 means the keypoint is undefined. 1 means it is defined.
        """
        x = annotation["keypoints"][::3]
        y = annotation["keypoints"][1::3]
        undef_ids = ((x > 0) & (y > 0)).astype(int)
        keypoints = []

        for pair in np.stack([x, y]).T:
            if pair[0] != -1:
                keypoints.append((pair[0], pair[1]))
            else:
                keypoints.append((0, 0))
        return keypoints, undef_ids
