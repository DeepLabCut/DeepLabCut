"""
Test script for super animal adaptation
"""

import os, subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

basepath = os.path.dirname(os.path.realpath(__file__))
videoname = "m3v1mp4"
video =  os.path.join(
    basepath, "openfield-Pranav-2018-10-30", "videos", videoname + ".mp4"
)

print ('testing superanimal_topviewmouse')

from deeplabcut.modelzoo.apis import SpatiotemporalAdaptation

superanimal_name = 'superanimal_topviewmouse'

videotype = '.mp4'


adapter = SpatiotemporalAdaptation(video,
                                   superanimal_name,
                                   modelfolder = "adapt_topview",
                                   videotype = videotype)

adapter.before_adapt_inference(pcutoff = 0.1)
adapter.adaptation_training()
adapter.after_adapt_inference(pcutoff = 0.1)


superanimal_name = 'superanimal_quadruped'

adapter = SpatiotemporalAdaptation(video,
                                   superanimal_name,
                                   modelfolder = "adapt_quadruped",
                                   videotype = videotype)

adapter.before_adapt_inference(pcutoff = 0.1)
adapter.adaptation_training()#pseudo_threshold = 0.0)
adapter.after_adapt_inference(pcutoff = 0.1)
