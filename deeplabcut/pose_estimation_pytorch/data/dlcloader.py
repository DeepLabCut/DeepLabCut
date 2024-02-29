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

import logging
import os
import pickle
from dataclasses import dataclass

import pandas as pd

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.helper import CombinedPropertyMeta
from deeplabcut.pose_estimation_pytorch.utils import df_to_generic
from deeplabcut.utils.auxiliaryfunctions import (
    get_bodyparts,
    get_model_folder,
    get_unique_bodyparts,
)


@dataclass
class DLCLoader(Loader, metaclass=CombinedPropertyMeta):
    """A Loader for DeepLabCut projects"""

    project_root: str
    model_config_path: str
    shuffle: int = 0
    image_id_offset: int = 0
    # TODO: read train fraction index

    properties = {
        "cfg": (
            deeplabcut.auxiliaryfunctions.read_config,
            lambda self: os.path.join(self.project_root, "config.yaml"),
        ),
        "model_folder": (
            lambda x: os.path.join(
                x[0], get_model_folder(x[1], x[2], x[3], engine=Engine.PYTORCH)
            ),
            lambda self: (
                self.project_root,
                self.cfg["TrainingFraction"][0],
                self.shuffle,
                self.cfg,
            ),
        ),
        "_datasets_folder": (
            lambda x: os.path.join(
                x[0], deeplabcut.auxiliaryfunctions.get_training_set_folder(x[1])
            ),
            lambda self: (self.project_root, self.cfg),
        ),
        "_path_dlc_data": (
            lambda x: os.path.join(x[0], f"CollectedData_{x[1]}.h5"),
            lambda self: (self._datasets_folder, self.cfg["scorer"]),
        ),
        "_path_dlc_doc": (
            lambda x: os.path.join(
                x[0], f"Documentation_data-{x[1]}_{x[2]}shuffle{x[3]}.pickle"
            ),
            lambda self: (
                self._datasets_folder,
                self.cfg["Task"],
                int(100 * self.cfg["TrainingFraction"][0]),
                self.shuffle,
            ),
        ),
    }

    def __post_init__(self):
        super().__init__(self.project_root, self.model_config_path)
        self.split, self.df_dlc, self.df_train, self.df_test = self._load_dlc_data()
        self.with_identity = self.has_identity_head(self.model_cfg)

    def get_dataset_parameters(self) -> PoseDatasetParameters:
        """
        Retrieves dataset parameters based on the instance's configuration.

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

    def _load_dlc_data(self):
        split = self._load_split(self._path_dlc_doc)
        df_dlc = pd.read_hdf(self._path_dlc_data)
        df_train, df_test = self.split_data(df_dlc, split)
        df_dlc, df_train, df_test = self.drop_duplicates(df_dlc, df_train, df_test)

        return split, df_dlc, df_train, df_test

    @staticmethod
    def drop_duplicates(dlc_df, df_train, df_test):
        dlc_df = dlc_df[~dlc_df.index.duplicated(keep="first")]
        df_train = df_train[~df_train.index.duplicated(keep="first")]
        if df_test is not None:
            df_test = df_test[~df_test.index.duplicated(keep="first")]
        return dlc_df, df_train, df_test

    def load_data(self, mode: str = "train") -> dict:
        """Loads DeepLabCut data into COCO-style annotations

        This function reads data from h5 file, split the data and returns it in
        COCO-like format

        Args:
            mode: mode indicating whether to use 'train' or 'test' data. Defaults to "train".

        Raises:
            AttributeError: if the specified mode (train or test) does not exist.

        Returns:
            the coco-style annotations
        """
        if mode == "train":
            data_dlc_format = self.df_train
        elif mode == "test":
            data_dlc_format = self.df_test
        # to do: add validation
        else:
            raise AttributeError(f"Unknown mode: {mode}")

        data = df_to_generic(self.project_root, data_dlc_format, self.image_id_offset)
        annotations_with_bbox = self._get_all_bboxes(
            data["images"], data["annotations"]
        )
        data["annotations"] = annotations_with_bbox

        return data

    @staticmethod
    def _load_split(path_dlc_doc: str) -> dict[str, list[int]]:
        """Summary:
        Split the annotation dataframe into train and test dataframes based on project's split
            that is downloaded from the project's directory

        Args:
            path_dlc_doc: the path to the DLC documentation file

        Return:
            the {"train": [train_ids], "test": [test_ids]} data split
        """
        with open(path_dlc_doc, "rb") as f:
            meta = pickle.load(f)

        train_ids = [int(i) for i in meta[1]]
        test_ids = [int(i) for i in meta[2]]

        return {"train": train_ids, "test": test_ids}

    @staticmethod
    def split_data(
        dlc_df: pd.DataFrame, split: dict[str, list[int]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DeepLabCut DataFrame into train/test dataframes

        Args:
            dlc_df: the dataframe containing the labeled data
            split: the train/test indices

        Returns:
            df_train, df_test
        """
        train_images = dlc_df.index[split["train"]]
        df_train = dlc_df.loc[train_images]

        df_test = None
        if len(split["test"]) != 0:
            test_images = dlc_df.index[split["test"]]
            df_test = dlc_df.loc[test_images]

        return df_train, df_test

    @staticmethod
    def has_identity_head(pytorch_config: dict) -> bool:
        return "identity" in pytorch_config.get("model", {}).get("heads", {})
