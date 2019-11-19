"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os

if os.environ.get('DLClight', default=False) == 'True':
    #print("DLC loaded in light mode; you cannot use the relabeling GUI!")
    pass
else:
    from deeplabcut.refine_training_dataset.refinement import *
    from deeplabcut.refine_training_dataset.auxfun_drag import *

from deeplabcut.refine_training_dataset.outlier_frames import *
