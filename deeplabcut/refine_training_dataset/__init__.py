"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
"""

import os

if os.environ.get('DLClight', default=False) == 'True':
    print("DLC loaded in light mode; you cannot use the relabeling GUI!")
    pass
else:
    from deeplabcut.refine_training_dataset.refinement import *
    from deeplabcut.refine_training_dataset.auxfun_drag import *

from deeplabcut.refine_training_dataset.outlier_frames import *
