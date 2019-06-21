
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import PoseDataset

def create(cfg):
    data = PoseDataset(cfg)
    return data
