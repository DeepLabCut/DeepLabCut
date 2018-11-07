'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import os
from deeplabcut.pose_estimation_tensorflow.util.logging import *

if os.environ.get('DLClight', default=False) == 'True':
    pass
else:
    from deeplabcut.pose_estimation_tensorflow.util.visualize import *
