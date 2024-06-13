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
import os
import pickle

import numpy as np
import pandas as pd

from deeplabcut.modelzoo.generalized_data_converter.datasets.base import BasePoseDataset
from deeplabcut.utils import auxiliaryfunctions


class BaseDLCPoseDataset(BasePoseDataset):

    def __init__(self, proj_root, dataset_name, shuffle=1, modelprefix=""):
        super(BaseDLCPoseDataset, self).__init__()

        assert proj_root != None and dataset_name != None

        self.meta["dataset_name"] = dataset_name
        self.meta["proj_root"] = proj_root
        self.meta["shuffle"] = shuffle
        self.meta["modelprefix"] = modelprefix

        self.proj_root = proj_root

        if modelprefix:
            config_file = os.path.join(self.proj_root, modelprefix + "_config.yaml")
        else:
            config_file = os.path.join(self.proj_root, "config.yaml")

        cfg = auxiliaryfunctions.read_config(config_file)

        task = cfg["Task"]

        scorer = cfg["scorer"]

        datasets_folder = os.path.join(
            self.proj_root,
            auxiliaryfunctions.GetTrainingSetFolder(cfg),
        )

        self.datasets_folder = datasets_folder

        trainingFraction = int(cfg["TrainingFraction"][0] * 100)

        path_dlc_collected = os.path.join(datasets_folder, f"CollectedData_{scorer}.h5")

        path_dlc_document = os.path.join(
            datasets_folder,
            f"Documentation_data-{task}_{trainingFraction}shuffle{shuffle}.pickle",
        )

        df = pd.read_hdf(path_dlc_collected)

        self.dlc_df = df

        with open(path_dlc_document, "rb") as f:
            document_data = pickle.load(f)

        train_indices = document_data[1]
        # index 2 is test indices
        test_indices = document_data[2]

        train_images = df.index[train_indices]
        test_images = df.index[test_indices]

        self.dlc_images = np.hstack([train_images, test_images])

        df_train = df.loc[train_images]

        df_test = df.loc[test_images]

        self.coco_train = self._df2generic(df_train)

        offset = len(self.coco_train["images"])

        self.coco_test = self._df2generic(df_test, image_id_offset=offset)

        self.populate_generic()

    def _df2generic(self, df, image_id_offset=0):
        raise NotImplementedError()

    def populate_generic(self):

        self.generic_train_images = self.coco_train["images"]
        self.generic_test_images = self.coco_test["images"]
        self.generic_train_annotations = self.coco_train["annotations"]
        self.generic_test_annotations = self.coco_test["annotations"]

        self.meta["categories"] = self.coco_test["categories"][0]

        # to build maps for later analysis
        self._build_maps()

        print(f"Before checking trainset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_train_images, self.generic_train_annotations
        )

        print(f"Before checking testset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_test_images, self.generic_test_annotations
        )
