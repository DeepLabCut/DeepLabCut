import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseDataset
from .dlcproject import DLCProject


class PoseDataset(Dataset, BaseDataset):

    def __init__(self,
                 project: DLCProject,
                 transform: object = None,
                 mode: str = 'train'):
        """

        Parameters
        ----------
        project: see class Project (wrapper for DLC original project class)
        transform: transformation function:

            def transform(image, keypoints):
                return image, keypoints

        mode: 'train' or 'test'
            this parameter which dataframe parse from the Project (df_tran or df_test)

        """
        super().__init__()
        self.transform = transform
        self.project = project
        self.cfg = self.project.cfg
        self.shuffle = self.project.shuffle
        self.project.convert2dict(mode)
        self.dataframe = self.project.dataframe

        # We must dropna because self.project.images doesn't contain imgaes with no labels so it can produce an indexnotfound error
        # length is stored here to avoid repeating the computation
        self.length = self.dataframe.dropna(axis=0, how="all").shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self,
                    index: int):
        """

        Parameters
        ----------
        index: int
            ordered number of the item in the dataset
        Returns
        -------
        image: torch.FloatTensor \in [0, 255]
            Tensor for the image from the dataset
        keypoints: list of keypoints

        train_dataset = PoseDataset(project, transform=transform)
        im, keypoints = train_dataset[0]

        """
        # load images
        try:
            image_file = self.dataframe.index[index]
            if isinstance(image_file, tuple):
                image_file = os.path.join(self.cfg["project_path"], *image_file)
            else:
                image_file = os.path.join(self.cfg["project_path"], image_file)
        except:
            print(len(self.project.images))
            print(index)
        image = cv2.imread(image_file)

        # load annotation
        annotation_index = self.project.image_path2index[image_file]
        annotation = self.project.annotations[annotation_index]
        keypoints, undef_ids = self.project.annotation2keypoints(annotation)

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            transformed['keypoints'] = [(-1, -1) if i in undef_ids else keypoint
                                        for i, keypoint in enumerate(transformed['keypoints'])]
        else:
            transformed = {}
            transformed['keypoints'] = keypoints
            transformed['image'] = image

        image = torch.FloatTensor(transformed['image']).permute(2, 0, 1)  # channels first

        assert len(transformed['keypoints']) == len(keypoints)
        keypoints = np.array(transformed['keypoints'])

        return image, keypoints
