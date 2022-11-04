import albumentations as A
import numpy as np
import pytest

import deeplabcut.pose_estimation_pytorch as dlc

def _get_dataset(path, transform):
    dlc_project = dlc.Project(path)
    dlc_project.train_test_split()
    dataset = dlc.PoseDataset(dlc_project,
                              transform=transform)
    return dataset


@pytest.mark.parametrize('path', ['/mnt/md0/shaokai/DLC-ModelZoo/data/all_topview/openfield-Pranav-2018-08-20'])
def test_check_train_test(path):
    dlc_project = dlc.Project(path)
    dlc_project.train_test_split()
    assert getattr(dlc_project, 'df_train', None) is not None
    assert getattr(dlc_project, 'df_test', None) is not None


@pytest.mark.parametrize('path', ['/mnt/md0/shaokai/DLC-ModelZoo/data/all_topview/openfield-Pranav-2018-08-20'])
def test_resize_transform(path):
    transform = A.Compose([
        A.Resize(width=256, height=256), ],
        keypoint_params=A.KeypointParams(format='xy'))

    dlc_project = dlc.Project(path)
    dlc_project.train_test_split()
    dataset = dlc.PoseDataset(dlc_project, transform=None)
    dataset_resized = dlc.PoseDataset(dlc_project, transform=transform)
    image_tensor_resized, keypoints_resized = dataset_resized[0]
    image_tensor, keypoints = dataset[0]

    assert image_tensor_resized.shape == (3, transform.transforms[0].height, transform.transforms[0].width)

    x_scale = image_tensor.shape[2] / image_tensor_resized.shape[2]
    y_scale = image_tensor.shape[1] / image_tensor_resized.shape[1]

    x_scale_keypoints = keypoints[:, 0] / keypoints_resized[:, 0]
    y_scale_keypoints = keypoints[:, 1] / keypoints_resized[:, 1]

    assert np.allclose(x_scale_keypoints, x_scale, atol=1e-4)
    assert np.allclose(y_scale_keypoints, y_scale, atol=1e-4)