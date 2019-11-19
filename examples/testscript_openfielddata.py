#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:06:13 2018

@author: alex

This is a test script that should achieve a certain performance on your system.

See values on our system, the DLC 2.0 docker with TF 1.8 on a NVIDIA GTX 1080Ti.
https://github.com/MMathisLab/Docker4DeepLabCut2.0

This test trains on the open field data set for about 30 minutes (15k iterations).

The results will be something like this:

Results for 15001  training iterations: 95 1 train error: 2.89 pixels. Test error: 2.81  pixels.
With pcutoff of 0.1  train error: 2.89 pixels. Test error: 2.81 pixels

The analysis of the video takes 41 seconds (batch size 32) and creating the frames 8 seconds (+ a few seconds for ffmpeg) to create the video.
"""

# Importing the toolbox (takes several seconds)
import deeplabcut
import os
from pathlib import Path

# Loading example data set
path_config_file = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30/config.yaml')
#deeplabcut.load_demo_data(path_config_file)

#shuffle=11 #>> imageio functions!
shuffle=12

deeplabcut.create_training_dataset(path_config_file,Shuffles=[shuffle])
cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)
posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(shuffle),'train/pose_cfg.yaml')

DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
DLC_config['save_iters']=10
DLC_config['display_iters']=2
DLC_config['multi_step']=[[0.005,15001]]
deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)


print("TRAIN NETWORK")
deeplabcut.train_network(path_config_file, shuffle=shuffle,saveiters=15000,displayiters=1000,max_snapshots_to_keep=15)

print("EVALUATE")
deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle],plotting=True)

#print("Analyze Video")
#videofile_path = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30','videos','m3v1mp4.mp4')
#deeplabcut.analyze_videos(path_config_file,[videofile_path], shuffle=shuffle)

#print("Create Labeled Video")
#deeplabcut.create_labeled_video(path_config_file,[videofile_path],save_frames=False, shuffle=shuffle)
