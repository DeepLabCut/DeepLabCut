import os
import pickle
from typing import List

import numpy as np
import pandas as pd

import deeplabcut
from .base import BaseProject
from ..utils import df2generic


class DLCProject(BaseProject):
    """
    Wrapper around the project containing information about the data, 
    the actual annotations and the configs

    Methods:
        - convert2dict : convert the annotations dataframe into a coco format dict of annotations
        - _init_annotation_image_correspondance: binds the image paths to corresponding annotations
                ensures there is no indexing offsets between images and annotations when
                going through the dataset
        - load_split : split the annotation dataframe into train and test dataframes 
                        based on project's split
        - annotation2keypoints : convert the coco annotations into array of keypoints
                                also returns the array of the keypoints' visibility
    """

    def __init__(self, proj_root:str,
                 shuffle: int = 0,
                 image_id_offset: int = 0,
                 keys_to_load: List[str] = ['images', 'annotations']):
        super().__init__()
        self.proj_root = proj_root
        self.shuffle = shuffle
        self.keys_to_load = keys_to_load
        self.image_id_offset = image_id_offset
        config_file = os.path.join(self.proj_root, 'config.yaml')
        self.cfg = deeplabcut.auxiliaryfunctions.read_config(config_file)
        self.task = self.cfg['Task']
        self.scorer = self.cfg['scorer']
        self.datasets_folder = os.path.join(
            self.proj_root, deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(self.cfg),
        )
        tr_frac = int(self.cfg['TrainingFraction'][0] * 100)
        self.path_dlc_data = os.path.join(self.datasets_folder, f'CollectedData_{self.scorer}.h5')
        self.path_dlc_doc = os.path.join(self.datasets_folder,
                                         f'Documentation_data-{self.task}_{tr_frac}shuffle{self.shuffle}.pickle')
        self.dlc_df = pd.read_hdf(self.path_dlc_data)
        self.load_split()
        self.dlc_df = self.dlc_df[~self.dlc_df.index.duplicated(keep = 'first')]
        self.df_train = self.df_train[~self.df_train.index.duplicated(keep = 'first')]
        if hasattr(self, "df_test"):
            self.df_test = self.df_test[~self.df_test.index.duplicated(keep = 'first')]

    def convert2dict(self,
                      mode: str = 'train'):
        """

        Parameters
        ----------
        mode

        Returns
        -------

        """
        try:
            self.dataframe = getattr(self, f'df_{mode}')
        except:
            raise AttributeError(f"PoseDataset doesn't have df_{mode} attr. Do project.train_test_split() first!")

        data = df2generic(self.proj_root,
                          self.dataframe,
                          self.image_id_offset)
        
        self._init_annotation_image_correspondance(data)

        for key in self.keys_to_load:
            setattr(self, key, data[key])
        print('The data has been loaded!')

    def _init_annotation_image_correspondance(self, data:dict):
        """data should be a COCO like dictionnary of the pose dataset"""
        
        # Path to id correspondance
        self.image_path2image_id = {}
        for i, image in enumerate(data["images"]):
            image_path = image["file_name"]
            self.image_path2image_id[image_path] = image['id']

        # id to annotations list
        self.id2annotations_idx = {}
        for i, annotation in enumerate(data['annotations']):
            image_id = annotation['image_id']
            try:
                self.id2annotations_idx[image_id].append(i)
            except KeyError:
                self.id2annotations_idx[image_id] = [i]

        return

    def load_split(self):
        """

        Returns
        -------

        """
        with open(self.path_dlc_doc, 'rb') as f:
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
    def annotation2keypoints(annotation):
        """
        TODO
        This function was copied from modelzoo project (transformation to coco format)
        Parameters
        ----------
        annotation: dict of annotations

        Returns
        -------
        keypoints: list
            paired keypoints
        undef_ids: array
            0 means this keypoints is undefined, 1 means it is
        """
        x = annotation['keypoints'][::3]
        y = annotation['keypoints'][1::3]
        undef_ids = ((x > 0) & (y > 0)).astype(int)
        keypoints = []

        for pair in np.stack([x, y]).T:
            if pair[0] != -1:
                keypoints.append((pair[0], pair[1]))
            else:
                keypoints.append((0, 0))
        return keypoints, undef_ids
