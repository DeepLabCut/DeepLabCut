import numpy as np
from enum import Enum


class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    pairwise_targets = 5
    pairwise_mask = 6
    data_item = 7


class DataItem:
    pass


def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)


def mirror_joints_map(all_joints, num_joints):
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def crop_image(joints, im, Xlabel, Ylabel, cfg):
    """Randomly cropping image around xlabel,ylabel taking into account size of image.
    Introduced in DLC 2.0 (Nature Protocols paper)"""
    widthforward = int(cfg["minsize"] + np.random.randint(cfg["rightwidth"]))
    widthback = int(cfg["minsize"] + np.random.randint(cfg["leftwidth"]))
    hup = int(cfg["minsize"] + np.random.randint(cfg["topheight"]))
    hdown = int(cfg["minsize"] + np.random.randint(cfg["bottomheight"]))
    Xstart = max(0, int(Xlabel - widthback))
    Xstop = min(np.shape(im)[1] - 1, int(Xlabel + widthforward))
    Ystart = max(0, int(Ylabel - hdown))
    Ystop = min(np.shape(im)[0] - 1, int(Ylabel + hup))
    joints[0, :, 1] -= Xstart
    joints[0, :, 2] -= Ystart

    inbounds = np.where(
        (joints[0, :, 1] > 0)
        * (joints[0, :, 1] < np.shape(im)[1])
        * (joints[0, :, 2] > 0)
        * (joints[0, :, 2] < np.shape(im)[0])
    )[0]
    return joints[:, inbounds, :], im[Ystart : Ystop + 1, Xstart : Xstop + 1, :]
