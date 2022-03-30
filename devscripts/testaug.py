# Script to test various augmentation related functions in DLC to understand how they work and improve documentation.
# Jeffrey Koppanyi - 18 Mar 22
from numpy import ndarray

# from utils import mirror_joints_map
from deeplabcut.utils import auxiliaryfunctions
import numpy as np


# num_joints = 6

poseconfig_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1PbOrHNSZe-6LS3ZxrCNCItYz2MSYtp9O/GET/DLCprojects/ppt5-Rachel-2022-01-17/dlc-models/iteration-0/ppt5Jan17-trainset95shuffle1/train/pose_cfg.yaml'
config_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1PbOrHNSZe-6LS3ZxrCNCItYz2MSYtp9O/GET/DLCprojects/ppt5-Rachel-2022-01-17/config.yaml'


def mirror_joints_map(all_joints, num_joints):
    res = np.arange(num_joints)

    symmetric_joints = []
    for p in all_joints:
        if len(p) == 2:
            symmetric_joints.append(p)
    # symmetric_joints = [p for p in all_joints if len(p) == 2]

    for pair in symmetric_joints:
        print(f"Res: {res}")
        print(f"pair: {pair}")
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def mirror_joint_coords(joints: np.ndarray, image_width: int) -> np.ndarray:
    # horizontally flip the x-coordinate, keep y unchanged
    joints[:, 1] = image_width - joints[:, 1] - 1
    return joints


def mirror_joints(joints: np.ndarray, symmetric_joints: dict, image_width: int) -> np.ndarray:
    # joint ids are 0 indexed
    res = np.copy(joints)
    res = mirror_joint_coords(res, image_width)
    # swap the joint_id for a symmetric one
    joint_id = joints[:, 0].astype(int)
    res[:, 0] = symmetric_joints[joint_id]
    return res


def main():

    # get parameters from train/pose_cfg.yaml
    pose_cfg = auxiliaryfunctions.read_config(poseconfig_path)
    # print(pose_cfg)

    all_joints = pose_cfg['all_joints']  # list of lists containing single integers - bodypart indices
    all_joints_names = pose_cfg['all_joints_names']  # list of strings - bodypart labels
    num_joints = pose_cfg['num_joints']  # int - number of bodyparts
    mirror = pose_cfg['mirror']  # boolean flag associated with mirroring the dataset

    # all_joints = [[0, 1], [2, 3], [4], [5], [6, 7], [8], [9], [10]]
    # all_joints_names = ['RH1', 'LH1', 'RH2', 'LH2', 'RH3', 'LH3', 'RH4', 'LH4', 'CRO', 'CRE', 'TTO']
    # num_joints = 11  # 8 + 3 = 11
    # mirror = True

    print(f"All: {all_joints}")
    print(f"All names: {all_joints_names}")
    print(f"Num: {num_joints}")
    print(f"Mirror: {mirror}")

    mirror_map = mirror_joints_map(all_joints, num_joints)

    print(f"All Joints: {all_joints}")
    print(f"N Joints: {num_joints}")
    print(f"Mapped Joints: {mirror_map}")

    print("mirror     normal")
    for name, mirror_idx in zip(all_joints_names, mirror_map):
        print(f"{all_joints_names[mirror_idx]} {name}")


    # # TODO joints needs to be a 2D array obtained from CSV. Idk how this is usually done..
    # # I need to understand how pose_deterministic is accessed and used.
    # mirror_coords = mirror_joints(joints, mirror_map, 1000)
    # print(mirror_coords)



if __name__ == '__main__':
    main()
