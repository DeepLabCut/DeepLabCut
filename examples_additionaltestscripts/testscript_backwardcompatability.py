#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:56:11 2018
@author: alex

DEVELOPERS:
This script tests various functionalities in an automatic way.

It should take about 3:30 minutes to run this in a CPU.
It should take about 1:30 minutes on a GPU (incl. downloading the ResNet weights)

It produces nothing of interest scientifically.
"""

task='TEST' # Enter the name of your experiment Task
scorer='Alex' # Enter the name of the experimenter/labeler


import os,  subprocess, deeplabcut
from pathlib import Path
import pandas as pd
import numpy as np

print("Imported DLC!")

'''
How to test this?

Put a link to one 'old DLC' project <2.1. Run the script and test:

- If video was not analyzed >> everything is stored with new DLC name
- If video was analzyed with old DLC name >> everything should *inherit* the old name!
- Same applies for evalute (should not be re-evaluated if it was evaluated...)
- If results exist (with previous DLC scorer name), they should not be recomputed!

'''

dfolder=None
net_type='resnet_50' #'mobilenet_v2_0.35' #'resnet_50'
augmenter_type='default' #'tensorpack'
numiter=10

'''
path_config_file='/home/alex/Hacking/Projects4TESTING/openfield-Pranav-2018-10-30/config.yaml'
newvideo='/home/alex/Hacking/Projects4TESTING/openfield-Pranav-2018-10-30/videos'

print("EVALUATE")
#deeplabcut.evaluate_network(path_config_file,plotting=True)

deeplabcut.analyze_videos(path_config_file,[newvideo],save_as_csv=True, destfolder=dfolder, dynamic=(True,.1,5),videotype='.mp4')

print("CREATE VIDEO")
deeplabcut.create_labeled_video(path_config_file,[newvideo], destfolder=dfolder,videotype='.mp4')

print("Making plots")
deeplabcut.plot_trajectories(path_config_file,[newvideo], destfolder=dfolder,videotype='.mp4')

print("Extracting skeleton distances, filter and plot filtered output")
deeplabcut.analyzeskeleton(path_config_file, [newvideo], save_as_csv=True, destfolder=dfolder,videotype='.mp4')

deeplabcut.filterpredictions(path_config_file,[newvideo],videotype='.mp4')
'''
###############################################################
## Testing backwards compatibility for 3D
###############################################################

path_config_file='/home/alex/Hacking/DeepLabCut/examples/TEST3D-Alex-2019-10-05-3d/config.yaml'
video_dir='/home/alex/Hacking/DeepLabCut/examples/3Dtestviews_videos'

print("TRIANGULATING")
deeplabcut.triangulate(path_config_file,video_dir,save_as_csv=True)

print("CREATING LABELED VIDEO 3-D")
deeplabcut.create_labeled_video_3d(path_config_file,[video_dir],start=5,end=10)
