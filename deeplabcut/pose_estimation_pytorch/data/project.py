import os
import pandas as pd
import deeplabcut
import pickle
import numpy as np

class Project:
    def __init__(self, proj_root,
                 shuffle = 1):
        self.proj_root = proj_root
        self.shuffle = shuffle

        config_file = os.path.join(self.proj_root, 'config.yaml')
        self.cfg = deeplabcut.auxiliaryfunctions.read_config(config_file)
        self.task = self.cfg['Task']
        self.scorer = self.cfg['scorer']
        self.datasets_folder = os.path.join(
            self.proj_root, deeplabcut.auxiliaryfunctions.GetTrainingSetFolder(self.cfg),
        )
        tr_frac = int(self.cfg['TrainingFraction'][0]*100)
        self.path_dlc_data = os.path.join(self.datasets_folder,f'CollectedData_{self.scorer}.h5')
        self.path_dlc_doc = os.path.join(self.datasets_folder,f'Documentation_data-{self.task}_{tr_frac}shuffle{self.shuffle}.pickle')
        self.dlc_df = pd.read_hdf(self.path_dlc_data)

    def train_test_split(self):
        with open(self.path_dlc_doc, 'rb') as f:
            meta = pickle.load(f)

        train_ids = meta[1]
        test_ids = meta[2]

        train_images = self.dlc_df.index[train_ids]
        test_images = self.dlc_df.index[test_ids]
        self.dlc_images = np.hstack([train_images,test_images])
        self.df_train = self.dlc_df.loc[train_images]
        self.df_test = self.dlc_df.loc[test_images]