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

from deeplabcut.pose_estimation_tensorflow.dataset import *
from deeplabcut.pose_estimation_tensorflow.models import *
from deeplabcut.pose_estimation_tensorflow.nnet import *
from deeplabcut.pose_estimation_tensorflow.util import *

from deeplabcut.pose_estimation_tensorflow.config import *
from deeplabcut.pose_estimation_tensorflow.default_config import *
from deeplabcut.pose_estimation_tensorflow.evaluate import *
from deeplabcut.pose_estimation_tensorflow.predict_videos import *
from deeplabcut.pose_estimation_tensorflow.test import *
from deeplabcut.pose_estimation_tensorflow.train import *
from deeplabcut.pose_estimation_tensorflow.training import *

import os
if os.environ.get('DLClight', default=False) == 'True':
    pass
else:
    from deeplabcut.pose_estimation_tensorflow.vis_dataset import *
