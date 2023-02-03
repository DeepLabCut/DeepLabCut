import numpy as np
import pandas as pd
import torch
from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
from deeplabcut.pose_estimation_tensorflow.core.predict import multi_pose_predict, argmax_pose_predict
from torch import nn
from typing import List


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


def get_scores(cfg,
               prediction: pd.DataFrame,
               target: pd.DataFrame,
               bodyparts: List = None):
    if cfg.get('pcutoff'):
        pcutoff = cfg['pcutoff']
        rmse, rmse_p = get_rmse(prediction, target, pcutoff,
                                bodyparts = bodyparts)
    else:
        rmse, rmse_p = get_rmse(prediction, target,
                                bodyparts = bodyparts)

    return np.nanmean(rmse), np.nanmean(rmse_p)


def get_rmse(prediction,
             target: pd.DataFrame,
             pcutoff: int=-1,
             bodyparts: List[str] =None):
    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]
    mask = prediction[scorer_pred].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts:
        diff = (target[scorer_target][bodyparts] - prediction[scorer_pred][bodyparts]) ** 2
    else:
        diff = (target[scorer_target] - prediction[scorer_pred]) ** 2
    mse = diff.xs("x", level=1, axis=1) + diff.xs("y", level=1, axis=1)
    rmse = np.sqrt(mse)
    rmse_p = np.sqrt(mse[mask])

    return rmse, rmse_p