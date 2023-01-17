import numpy as np
import torch
from torch import nn

from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
from deeplabcut.pose_estimation_tensorflow.core.predict import multi_pose_predict, argmax_pose_predict


def get_prediction(cfg, output, stride=8):
    '''
    get predictions from model output
    output = heatmaps, locref
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    locref: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    poses = []
    heatmaps, locref = output
    heatmaps = nn.Sigmoid()(heatmaps)
    heatmaps = heatmaps.permute(0, 2, 3, 1).detach().cpu().numpy()
    locref = locref.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(heatmaps.shape[0]):
        shape = locref[i].shape
        locref_i = np.reshape(locref, (shape[0], shape[1], -1, 2))
        if cfg['location_refinement']:
            locref_i = locref_i * cfg['locref_stdev']
        pose = multi_pose_predict(heatmaps[i], locref_i, stride, 1)
        poses.append(pose)
    return np.stack(poses, axis=0)
