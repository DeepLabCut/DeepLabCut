import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from deeplabcut.utils.auxiliaryfunctions import read_plainconfig, get_model_folder
from deeplabcut.pose_estimation_pytorch.models.target_generators import BaseGenerator

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
        self.num_joints = len(self.cfg['bodyparts'])
        self.shuffle = self.project.shuffle
        self.project.convert2dict(mode)
        self.dataframe = self.project.dataframe

        modelfolder = os.path.join(
            self.project.proj_root,
            get_model_folder(
                self.cfg['TrainingFraction'][0],
                self.shuffle,
                self.cfg,
                '',
            )
        )
        pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
        pytorch_cfg = read_plainconfig(pytorch_config_path)
        self.with_center = pytorch_cfg.get('with_center', False)
        self.max_num_animals = len(self.cfg.get('individuals', ['0']))

        # We must dropna because self.project.images doesn't contain imgaes with no labels so it can produce an indexnotfound error
        # length is stored here to avoid repeating the computation
        self.length = self.dataframe.dropna(axis=0, how="all").shape[0]

    def __len__(self):
        return self.length
    
    def _calc_area_from_keypoints(self, keypoints):
        w = keypoints[:, :, 0].max(axis=1) - keypoints[:, :, 0].min(axis=1)
        h = keypoints[:, :, 1].max(axis=1) - keypoints[:, :, 1].min(axis=1)
        return w*h
    
    def _keypoint_in_boundary(self, keypoint, shape):
        '''
        
        Parameters
        ----------
            keypoint: [x, y]
            shape: (height, width)
        Returns
        -------
            bool : wether a keypoint lies inside the given shape'''

        return (keypoint[0] > 0) and (keypoint[1] > 0) and (keypoint[0] < shape[1]) and (keypoint[1] < shape[0])
    
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
        original_size = image.shape

        # load annotations
        image_id = self.project.image_path2image_id[image_file]
        n_annotations = len(self.project.id2annotations_idx[image_id])

        if not self.with_center:
            keypoints = np.zeros((self.max_num_animals, self.num_joints, 3))
            num_keypoints_returned = self.num_joints
        else:
            keypoints = np.zeros((self.max_num_animals, self.num_joints + 1, 3))
            num_keypoints_returned = self.num_joints + 1

        for i, annotation_idx in enumerate(self.project.id2annotations_idx[image_id]):
            _annotation = self.project.annotations[annotation_idx]
            _keypoints, _undef_ids = self.project.annotation2keypoints(_annotation)
            _keypoints = np.array(_keypoints)

            if self.with_center:
                keypoints[i, :-1, :2] = _keypoints
                keypoints[i, :-1, 2] = _undef_ids
                
            else:
                keypoints[i, :, :2] = _keypoints
                keypoints[i, :, 2] = _undef_ids


        # Needs two be 2 dimensional for albumentations
        keypoints = keypoints.reshape((-1, 3))

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints[:, :2])
            shape_transformed = transformed['image'].shape
            transformed['keypoints'] = [(-1, -1) 
                                        if ((keypoints[i, 2] == 0) or (not self._keypoint_in_boundary(keypoint, shape_transformed)))
                                        else keypoint
                                        for i, keypoint in enumerate(transformed['keypoints'])]
        else:
            transformed = {}
            transformed['keypoints'] = keypoints[:, :2]
            transformed['image'] = image

        image = torch.FloatTensor(transformed['image']).permute(2, 0, 1)  # channels first

        assert len(transformed['keypoints']) == len(keypoints)
        keypoints = np.array(transformed['keypoints']).reshape((n_annotations, num_keypoints_returned, 2))

        #TODO Quite ugly
        #      
        # Center keypoint needs to be computed after transformation because
        # it should depend on visible keypoints only (which may change after augmentation)
        if self.with_center:
            try:
                keypoints[:, -1, :] = keypoints[:, :-1, :][~np.any(keypoints[:, :-1, :] == -1, axis=2)].reshape(n_annotations, -1, 2).mean(axis = 1) 
            except ValueError:
                # For at least one annotation every keypoint is out of the frame
                for i in range(keypoints.shape[0]):
                    try:
                        keypoints[i, -1, :] = keypoints[i, :-1, :][~np.any(keypoints[i, :-1, :] == -1, axis=1)].mean(axis = 0)
                    except ValueError:
                        keypoints[i, -1, :] = np.array([-1, -1])

        np.nan_to_num(keypoints, copy=False, nan=-1)
        area = self._calc_area_from_keypoints(keypoints)


        res = {}
        res['image'] = image
        res['original_size'] = original_size # In order to convert back the keypoints to their original space
        res['annotations'] = {}
        res['annotations']['keypoints'] = keypoints
        res['annotations']['area'] = area


        return res
