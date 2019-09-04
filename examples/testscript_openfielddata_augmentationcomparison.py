#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is a test script to compare the loaders. tensorpack allows much more choices for augmentation. The parameters
can be set in pose_dataset_tensorpack.py and of course specifically in each pose_config.yaml file before training. In fact,
pose_dataset_tensorpack.py will fall back to default parameters if they are not defined in pose_config.yaml and one is
using dataset_type:'tensorpack'

This script creates one identical split for the openfield test dataset and trains it with the 
standard loader and the tensorpack loader for 50k iterations in DLC 2.0 docker with TF 1.8 on a NVIDIA GTX 1080Ti.

My results were:

**With standard loader:**

Training iterations:	%Training dataset	Shuffle number	 Train error(px)	 Test error(px)	p-cutoff used	Train error with p-cutoff	Test error with p-cutoff
10000	80	2	2.64	3.11	0.4	2.64	3.11
20000	80	2	2.26	2.72	0.4	2.26	2.72
30000	80	2	1.71	2.28	0.4	1.71	2.28
40000	80	2	1.88	2.61	0.4	1.88	2.61
50000	80	2	1.86	2.32	0.4	1.86	2.32
60000	80	2	1.92	2.42	0.4	1.92	2.42
70000	80	2	2.38	3.04	0.4	2.38	3.04
80000	80	2	1.55	2.34	0.4	1.55	2.34
90000	80	2	1.5	2.27	0.4	1.5	2.27
100000	80	2	1.52	2.34	0.4	1.52	2.34


**With tensorpack loader:**

Training iterations:	%Training dataset	Shuffle number	 Train error(px)	 Test error(px)	p-cutoff used	Train error with p-cutoff	Test error with p-cutoff
10000	80	3	2.35	2.91	0.4	2.35	2.91
20000	80	3	3.28	3.51	0.4	3.28	3.51
30000	80	3	1.57	2.24	0.4	1.57	2.24
40000	80	3	3.54	4.17	0.4	3.54	4.17
50000	80	3	1.76	2.74	0.4	1.76	2.74
60000	80	3	2.85	3.39	0.4	2.85	3.39
70000	80	3	3.88	4.71	0.4	3.88	4.71
80000	80	3	1.2	2.06	0.4	1.2	2.06
90000	80	3	2.2	3.07	0.4	2.2	3.07
100000	80	3	1.06	1.96	0.4	1.06	1.96




For details on TensorPack check out:

A Neural Net Training Interface on TensorFlow, with focus on speed + flexibility
https://github.com/tensorpack/tensorpack
"""


import os
from pathlib import Path
os.environ['DLClight']='True'

import deeplabcut
# Loading example data set
path_config_file = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30/config.yaml')
cfg=deeplabcut.auxiliaryfunctions.read_config(path_config_file)


deeplabcut.load_demo_data(path_config_file)
#create one split and make Shuffle 5 and 6 have the same split.
trainIndexes, testIndexes=deeplabcut.mergeandsplit(path_config_file,trainindex=0,uniform=True)
deeplabcut.create_training_dataset(path_config_file,Shuffles=[2],trainIndexes=trainIndexes,testIndexes=testIndexes) 
deeplabcut.create_training_dataset(path_config_file,Shuffles=[3],trainIndexes=trainIndexes,testIndexes=testIndexes)

for shuffle in [2,3]:
	if shuffle==3:
		posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(shuffle),'train/pose_cfg.yaml')

		DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
		DLC_config['dataset_type']='tensorpack'
		deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

	#otherwise default...
	print("TRAIN NETWORK", shuffle)
	deeplabcut.train_network(path_config_file, shuffle=shuffle,saveiters=10000,displayiters=200,maxiters=100000,max_snapshots_to_keep=11)

	print("EVALUATE")
	deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle],plotting=True)

	#DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
	#DLC_config['save_iters']=10
	#DLC_config['display_iters']=2
	#DLC_config['multi_step']=[[0.005,15001]]
	#deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)

	print("Analyze Video")

	videofile_path = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30','videos','m3v1mp4.mp4')

	deeplabcut.analyze_videos(path_config_file,[videofile_path], shuffle=shuffle)

	print("Create Labeled Video and plot")
	deeplabcut.create_labeled_video(path_config_file,[videofile_path], shuffle=shuffle)
	deeplabcut.plot_trajectories(path_config_file,[videofile_path], shuffle=shuffle)
