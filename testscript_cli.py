#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modified from: https://github.com/DeepLabCut/DeepLabCut-core/testscript_cli.py
by Mackenzie.

DEVELOPERS:
This script tests various functionalities in an automatic way.
It produces nothing of interest scientifically.
"""

task = "Testcore"  # Enter the name of your experiment Task
scorer = "Mackenzie"  # Enter the name of the experimenter/labeler

import os, subprocess, sys


# def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install("tensorflow==1.13.1")

import deeplabcut as dlc
from deeplabcut.core.engine import Engine

from pathlib import Path
import pandas as pd
import numpy as np
import platform

print("Imported DLC!")

engine = Engine.PYTORCH

basepath = os.path.dirname(os.path.abspath("testscript_cli.py"))
videoname = "reachingvideo1"
video = [
    os.path.join(
        basepath,
        "examples",
        "Reaching-Mackenzie-2018-08-30",
        "videos",
        videoname + ".avi",
    )
]
# For testing a color video:
# videoname='baby4hin2min'
# video=[os.path.join('/home/alex/Desktop/Data',videoname+'.mp4')]
# to test destination folder:
# dfolder=basepath
print(video)

dfolder = None
net_type = "resnet_50"  #'mobilenet_v2_0.35' #'resnet_50'
augmenter_type = "default"
augmenter_type2 = "imgaug"

if platform.system() == "Darwin" or platform.system() == "Windows":
    print("On Windows/OSX tensorpack is not tested by default.")
    augmenter_type3 = "imgaug"
else:
    augmenter_type3 = "tensorpack"  # Does not work on WINDOWS

numiter = 3

print("CREATING PROJECT")
path_config_file = dlc.create_new_project(task, scorer, video, copy_videos=True)

cfg = dlc.auxiliaryfunctions.read_config(path_config_file)
cfg["numframes2pick"] = 5
cfg["pcutoff"] = 0.01
cfg["TrainingFraction"] = [0.8]
cfg["skeleton"] = [["bodypart1", "bodypart2"], ["bodypart1", "bodypart3"]]

dlc.auxiliaryfunctions.write_config(path_config_file, cfg)

print("EXTRACTING FRAMES")
dlc.extract_frames(path_config_file, mode="automatic", userfeedback=False)

print("CREATING SOME LABELS FOR THE FRAMES")
frames = os.listdir(os.path.join(cfg["project_path"], "labeled-data", videoname))
# As this next step is manual, we update the labels by putting them on the diagonal (fixed for all frames)
for index, bodypart in enumerate(cfg["bodyparts"]):
    columnindex = pd.MultiIndex.from_product(
        [[scorer], [bodypart], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    frame = pd.DataFrame(
        100 + np.ones((len(frames), 2)) * 50 * index,
        columns=columnindex,
        index=[os.path.join("labeled-data", videoname, fn) for fn in frames],
    )
    if index == 0:
        dataFrame = frame
    else:
        dataFrame = pd.concat([dataFrame, frame], axis=1)

dataFrame.to_csv(
    os.path.join(
        cfg["project_path"],
        "labeled-data",
        videoname,
        "CollectedData_" + scorer + ".csv",
    )
)
dataFrame.to_hdf(
    os.path.join(
        cfg["project_path"],
        "labeled-data",
        videoname,
        "CollectedData_" + scorer + ".h5",
    ),
    "df_with_missing",
    format="table",
    mode="w",
)

print("Plot labels...")

dlc.check_labels(path_config_file)

print("CREATING TRAININGSET")
dlc.create_training_dataset(
    path_config_file, net_type=net_type, augmenter_type=augmenter_type, engine=engine,
)

print("TRAIN")
dlc.train_network(path_config_file, epochs=numiter, displayiters=2)

print("EVALUATE")
dlc.evaluate_network(path_config_file, plotting=True)

videotest = os.path.join(cfg["project_path"], "videos", videoname + ".avi")

print(videotest)

# quicker variant
"""
print("VIDEO ANALYSIS")
dlc.analyze_videos(path_config_file, [videotest], save_as_csv=True)

print("CREATE VIDEO")
dlc.create_labeled_video(path_config_file,[videotest], save_frames=False)

print("Making plots")
dlc.plot_trajectories(path_config_file,[videotest])

print("CREATING TRAININGSET 2")
dlc.create_training_dataset(path_config_file, Shuffles=[2],net_type=net_type,augmenter_type=augmenter_type2)

cfg=dlc.auxiliaryfunctions.read_config(path_config_file)
posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(2),'train/pose_cfg.yaml')
DLC_config=dlc.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=numiter
DLC_config['display_iters']=1
DLC_config['multi_step']=[[0.001,numiter]]

print("CHANGING training parameters to end quickly!")
dlc.auxiliaryfunctions.write_config(posefile,DLC_config)

print("TRAIN")
dlc.train_network(path_config_file, shuffle=2,allow_growth=True)

print("EVALUATE")
dlc.evaluate_network(path_config_file,Shuffles=[2],plotting=False)


print("ANALYZING some individual frames")
dlc.analyze_time_lapse_frames(path_config_file,os.path.join(cfg['project_path'],'labeled-data/reachingvideo1/'))
"""

print("Export model...")
dlc.export_model(path_config_file, shuffle=1, make_tar=False)

print(
    "ALL DONE!!! - default/imgaug cases of DLCcore training and evaluation are functional (no extract outlier or refinement tested)."
)
