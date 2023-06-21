import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils.auxiliaryfunctions import read_plainconfig, get_model_folder
from .base import BaseDataset
from .dlcproject import DLCProject


class PoseDataset(Dataset, BaseDataset):
    """
        Dataset for pose estimation
    """

    def __init__(self,
                 project: DLCProject,
                 transform: object = None,
                 mode: str = 'train'):
        """

        Parameters
        ----------
        project: see class Project (wrapper for DLC original project class)
        transform: augmentation/normalization pipeline

        mode: 'train' or 'test'
            this parameter which dataframe parse from the Project (df_tran or df_test)

        """
        super().__init__()
        self.transform = transform
        self.project = project
        self.cfg = self.project.cfg

        self.bodyparts = auxiliaryfunctions.get_bodyparts(self.cfg)
        self.num_joints = len(self.bodyparts)

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
        self.color_mode = pytorch_cfg.get('colormode', 'RGB')

        self.length = self.dataframe.shape[0]
        assert self.length == len(self.project.image_path2image_id.keys())

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
                    index: int) -> dict:
        """

        Parameters
        ----------
        index: int
            ordered number of the item in the dataset
        Returns
        -------
        dictionnary corresponding to the image, annotations...
            keys:
                -'image' : image
                -'annotations':
                    -'keypoints' : array of keypoints, invisible keypoints appear as (-1, -1)
                    -'area': array of animals area in this image
                -'original_size' : original size of the image before applying transforms
                                    useful to convert the predictions/ground truth back to
                                    the input space

        train_dataset = PoseDataset(project, transform=transform)
        pose_dict = train_dataset[0]

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
        if self.color_mode == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape

        # load annotations
        image_id = self.project.image_path2image_id[image_file]
        n_annotations = len(self.project.id2annotations_idx[image_id])

        bodyparts = [bpt for bpt in self.bodyparts]
        if not self.with_center:
            keypoints = np.zeros((self.max_num_animals, self.num_joints, 3))
            num_keypoints_returned = self.num_joints
        else:
            keypoints = np.zeros((self.max_num_animals, self.num_joints + 1, 3))
            num_keypoints_returned = self.num_joints + 1
            bodyparts += ["_center_"]

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
            class_labels = [
                f"individual{i}_{bpt}"
                for i in range(self.max_num_animals)
                for bpt in bodyparts
            ]
            transformed = self.transform(
                image=image,
                keypoints=keypoints[:, :2],
                class_labels=class_labels,
            )

            # Discard keypoints that aren't in the frame anymore
            shape_transformed = transformed['image'].shape
            transformed['keypoints'] = [
                keypoint if self._keypoint_in_boundary(keypoint, shape_transformed)
                else (-1, -1)
                for keypoint in transformed['keypoints']
            ]

            # Discard keypoints that are undefined
            undef_class_labels = [
                class_labels[i] for i, kpt in enumerate(keypoints) if kpt[2] == 0
            ]
            for label in undef_class_labels:
                new_index = transformed["class_labels"].index(label)
                transformed['keypoints'][new_index] = (-1, -1)

        else:
            transformed = {}
            transformed['keypoints'] = keypoints[:, :2]
            transformed['image'] = image

        image = torch.FloatTensor(transformed['image']).permute(2, 0, 1)  # channels first

        assert len(transformed['keypoints']) == len(keypoints)
        keypoints = np.array(transformed['keypoints']).reshape((n_annotations, num_keypoints_returned, 2)).astype(float)

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
