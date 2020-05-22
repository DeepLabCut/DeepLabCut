#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

ls Movie_Part_1.mp4 Movie_Part_2.mp4 | perl -ne 'print "file $_"' | ffmpeg -f concat -i - -c copy Movie_Joined.mp4

"""


import os, subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

path_config_file = (
    "/home/alex/Hacking/Projects4TESTING/openfield-Pranav-2018-10-30/config.yaml"
)
video = "/home/alex/Hacking/Projects4TESTING/openfield-Pranav-2018-10-30/vid/Movie_Joined.mp4"  # m3v1mp4.mp4'

# newvideo=deeplabcut.ShortenVideo(video,start='00:00:00',stop='00:00:20',outsuffix='short')

deeplabcut.analyze_videos(
    path_config_file, [video], save_as_csv=True, dynamic=(True, 0.1, 30)
)  # ,cropping=[0,200,0,200])

# print("CREATE VIDEO")
deeplabcut.create_labeled_video(path_config_file, [video])

"""
# Checked: dynamic=(True,1.1,0)) >> only keeps large limits (full frame >> never detects)
 dynamic=(True,.1,0)) >> box too small, looses all the time, but algo redoes analysis on full frame to recover!
deeplabcut.analyze_videos(path_config_file,[video],save_as_csv=True, dynamic=(True,.1,30),cropping=[0,200,0,200])
>> the mouse leaves the bounding box! All correct I.e. lost...


"""
