"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from deeplabcut import DEBUG
import os

if os.environ.get('DLClight', default=False) == 'True':
    #print("DLC loaded in light mode; you cannot use the labeling GUI!")
    pass
else:
    from deeplabcut.generate_training_dataset.auxfun_drag_label import *
    from deeplabcut.generate_training_dataset.labeling_toolbox import *
    from deeplabcut.generate_training_dataset.multiple_individuals_labeling_toolbox import *
    from deeplabcut.generate_training_dataset.frame_extraction_toolbox import *

from deeplabcut.generate_training_dataset.frame_extraction import *
from deeplabcut.generate_training_dataset.trainingsetmanipulation import *
