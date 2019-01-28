"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
"""
from deeplabcut import DEBUG
import os

if os.environ.get('DLClight', default=False) == 'True':
    print("DLC loaded in light mode; you cannot use the labeling GUI!")
    pass
else:
    from deeplabcut.generate_training_dataset.auxfun_drag_label import *
    from deeplabcut.generate_training_dataset.labeling_toolbox import *
    from deeplabcut.generate_training_dataset.frame_extraction_toolbox import *

from deeplabcut.generate_training_dataset.frame_extraction import *
from deeplabcut.generate_training_dataset.trainingsetmanipulation import *
