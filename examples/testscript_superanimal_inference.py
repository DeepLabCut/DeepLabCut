"""
Testscript for super animal inference

"""
import os, subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

basepath = os.path.dirname(os.path.realpath(__file__))
videoname = "reachingvideo1"
video = [
    os.path.join(
        basepath, "Reaching-Mackenzie-2018-08-30", "videos", videoname + ".avi"
    )
]


print ('testing superanimal_topviewmouse')
superanimal_name = 'superanimal_topviewmouse'
scale_list = [200, 300, 400]
deeplabcut.video_inference_superanimal(
    video,
    superanimal_name,
    videotype='avi',
    scale_list = scale_list,
)

print ('testing superanimal_topviewmouse')
superanimal_name = 'superanimal_quadruped'
deeplabcut.video_inference_superanimal(
    video,
    superanimal_name,
    videotype='avi',
    scale_list = scale_list,
)
