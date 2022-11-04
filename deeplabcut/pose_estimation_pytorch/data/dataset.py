import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from deeplabcut.pose_estimation_pytorch.data.base import Base
from deeplabcut.pose_estimation_pytorch.utils import df2generic


class PoseDataset(Dataset, Base):

    def __init__(self,
                 project,
                 transform=None,
                 image_id_offset=0,
                 mode='train'):
        """

        Parameters
        ----------
        project: see class Project (wrapper for DLC original project class)
        transform: transformation function:

            def transform(image, keypoints):
                return image, keypoints

        image_id_offset: TODO
        mode: 'train' or 'test'
            this parameter specify which dataframe parse from the Project (df_tran or df_test)

        """
        super().__init__()
        try:
            self.dataframe = getattr(project, f'df_{mode}')
        except:
            raise AttributeError(f"PoseDataset doesn't have df_{mode} attr. Do project.train_test_split() first!")

        data = df2generic(project.proj_root, self.dataframe, image_id_offset)

        self.images = data['images']
        self.keypoints = data['annotations']
        self.transform = transform
        self.cfg = project.cfg
        assert len(self.images) == len(self.keypoints)

    def create_from_config(self, config):
        # TODO
        pass

    @staticmethod
    def _annotation2key(annotation):
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
        undef_ids: list
            mask
        """
        x = annotation['keypoints'][::3]
        y = annotation['keypoints'][1::3]
        vis = annotation['keypoints'][2::3]
        undef_ids = list(np.where(x == -1)[0])
        keypoints = []

        for pair in np.stack([x, y]).T:
            if pair[0] != -1:
                keypoints.append((pair[0], pair[1]))
            else:
                keypoints.append((0, 0))
        return keypoints, undef_ids

    def __len__(self):
        return len(self.images)

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
        im, keipoints = train_dataset[0]

        """
        image_file = self.images[index]['file_name']
        image = cv2.imread(image_file)

        annotation = self.keypoints[index]
        keypoints, undef_ids = self._annotation2key(annotation)
        if self.transform:

            transformed = self.transform(image=image, keypoints=keypoints)
            transformed['keipoints'] = [(-1, -1) if i in undef_ids else keypoint
                                        for i, keypoint in enumerate(transformed['keypoints'])]
        else:
            transformed = {}
            transformed['keipoints'] = keypoints
            transformed['image'] = image

        image = torch.FloatTensor(transformed['image']).permute(2, 0, 1)  # channels first

        assert len(transformed['keipoints']) == len(keypoints)
        keypoints = np.array(transformed['keipoints'])

        return image, keypoints
