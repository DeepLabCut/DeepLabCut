"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

The feature detectors in DeepLabCut are based of DeeperCut.
Our toolbox provides an interface to DeeperCut by Eldar Insafutdinov et al.
https://github.com/eldar/pose-tensorflow

pose_estimation_tensorflow is based of Eldar's repository.
"""
from deeplabcut.pose_estimation_tensorflow.nnet.losses import *
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import *
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import *
from deeplabcut.pose_estimation_tensorflow.nnet.predict import *
