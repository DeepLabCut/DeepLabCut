"""TODO"""
import torch

import deeplabcut.pose_estimation_pytorch.models as dlc_models

def _get_keypoints(number_of_joints: int = 4,
                   axis: int = 2):
    keypoints_torch = torch.Tensor(number_of_joints, axis)

    return keypoints_torch, number_of_joints


def test_generate_heatmaps():

    keypoints_torch, number_of_joints = _get_keypoints()
    image_size = (256, 256)
    sigma = 5
    heatmap_size = (64, 64)
    heatmaps = dlc_models._generate_heatmaps(keypoints_torch,
                                                   heatmap_size,
                                                   image_size,
                                                   sigma=sigma)
    assert heatmaps.shape == (number_of_joints, heatmap_size[0], heatmap_size[1])