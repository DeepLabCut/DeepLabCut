'''
Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import PoseDataset

def create(cfg):
    data = PoseDataset(cfg)
    return data
