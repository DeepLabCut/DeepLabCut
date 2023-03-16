import numpy as np
import pandas as pd
import torch
from deeplabcut.pose_estimation_pytorch.models.utils import generate_heatmaps
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


def get_top_values(scmap, n_top=5):
    batchsize, ny, nx, num_joints = scmap.shape
    scmap_flat = scmap.reshape(batchsize, nx * ny, num_joints)
    if n_top == 1:
        scmap_top = np.argmax(scmap_flat, axis=1)[None]
    else:
        scmap_top = np.argpartition(scmap_flat, -n_top, axis=1)[:, -n_top:]
        for ix in range(batchsize):
            vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
        scmap_top = scmap_top.swapaxes(0, 1)

    Y, X = np.unravel_index(scmap_top, (ny, nx))
    return Y, X


def multi_pose_predict(scmap, locref, stride, num_outputs):
    Y, X = get_top_values(scmap[None], num_outputs)
    Y, X = Y[:, 0], X[:, 0]
    num_joints = scmap.shape[2]

    DZ = np.zeros((num_outputs, num_joints, 3))
    indices = np.indices((num_outputs, num_joints))
    x = X[indices[0], indices[1]]
    y = Y[indices[0], indices[1]]
    DZ[:, :, :2] = locref[y, x, indices[1], :]
    DZ[:, :, 2] = scmap[y, x, indices[1]]

    X = X.astype("float32") * stride[1] + 0.5 * stride[1] + DZ[:, :, 0]
    Y = Y.astype("float32") * stride[0] + 0.5 * stride[0] + DZ[:, :, 1]
    P = DZ[:, :, 2]

    pose = np.empty((num_joints, num_outputs * 3), dtype="float32")
    pose[:, 0::3] = X.T
    pose[:, 1::3] = Y.T
    pose[:, 2::3] = P.T

    return pose

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
             pcutoff: float=-1,
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