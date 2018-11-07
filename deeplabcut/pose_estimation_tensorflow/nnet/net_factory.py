'''
Adopted: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import PoseNet


def pose_net(cfg):
    cls = PoseNet
    return cls(cfg)
