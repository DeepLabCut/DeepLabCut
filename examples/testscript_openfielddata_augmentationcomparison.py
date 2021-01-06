#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is a test script to compare the loaders. tensorpack allows much more choices for augmentation. The parameters
can be set in pose_dataset_tensorpack.py and of course specifically in each pose_config.yaml file before training. In fact,
pose_dataset_tensorpack.py will fall back to default parameters if they are not defined in pose_config.yaml and one is
using dataset_type:'tensorpack'

This script creates one identical split for the openfield test dataset and trains it with the
standard loader and the tensorpack loader for k iterations in DLC 2.0 docker with TF 1.8 on a NVIDIA GTX 1080Ti.

My results were (Run with DLC 2.0.9 in Sept 2019)

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

My results were (Run with DLC 2.2b5 in May 2020) for 20k iterations

Imagaug augmentation:

Results for 20000  training iterations: 95 1 train error: 3.25 pixels. Test error: 4.98  pixels.
With pcutoff of 0.4  train error: 3.25 pixels. Test error: 4.98 pixels

Default augmentation:

Results for 20000  training iterations: 95 2 train error: 2.5 pixels. Test error: 4.08  pixels.
With pcutoff of 0.4  train error: 2.5 pixels. Test error: 4.08 pixels

Tensorpack augmentation:

Results for 20000  training iterations: 95 3 train error: 3.06 pixels. Test error: 4.78  pixels.
With pcutoff of 0.4  train error: 3.06 pixels. Test error: 4.78 pixels

My results were (Run with DLC *2.2b7* in July 2020) for 20k iterations

Attention: default changed!

***Default = Imagaug**** augmentation:

Done and results stored for snapshot:  snapshot-20000
Results for 20000  training iterations: 95 1 train error: 2.93 pixels. Test error: 3.09  pixels.
With pcutoff of 0.4  train error: 2.93 pixels. Test error: 3.09 pixels

Scalecrop (was = default) augmentation:

Done and results stored for snapshot:  snapshot-20000
Results for 20000  training iterations: 95 2 train error: 2.5 pixels. Test error: 2.57  pixels.
With pcutoff of 0.4  train error: 2.5 pixels. Test error: 2.57 pixels

Tensorpack augmentation:

Done and results stored for snapshot:  snapshot-20000
Results for 20000  training iterations: 95 3 train error: 3.1 pixels. Test error: 3.29  pixels.
With pcutoff of 0.4  train error: 3.1 pixels. Test error: 3.29 pixels

My results were (Run with DLC *2.2b7* on August 1st 2020) for 10k iterations

Imgaug:
Results for 10000  training iterations: 95 1 train error: 3.78 pixels. Test error: 3.89  pixels.
With pcutoff of 0.4  train error: 3.78 pixels. Test error: 3.89 pixels

Scalecrop:
Done and results stored for snapshot:  snapshot-10000
Results for 10000  training iterations: 95 2 train error: 2.81 pixels. Test error: 2.46  pixels.
With pcutoff of 0.4  train error: 2.81 pixels. Test error: 2.46 pixels

Tensorpack:
Done and results stored for snapshot:  snapshot-10000
Results for 10000  training iterations: 95 3 train error: 3.76 pixels. Test error: 3.98  pixels.
With pcutoff of 0.4  train error: 3.76 pixels. Test error: 3.98 pixels


My results were (Run with DLC *2.2b8* on Sept 7 2020) for 10k iterations

Imgaug:
Results for 10000  training iterations: 95 1 train error: 2.63 pixels. Test error: 3.88  pixels.
With pcutoff of 0.4  train error: 2.63 pixels. Test error: 3.88 pixels

Scalecrop:
Results for 10000  training iterations: 95 2 train error: 3.08 pixels. Test error: 4.02  pixels.
With pcutoff of 0.4  train error: 3.08 pixels. Test error: 4.02 pixels

Tensorpack:
Results for 10000  training iterations: 95 3 train error: 2.9 pixels. Test error: 3.31  pixels.
With pcutoff of 0.4  train error: 2.9 pixels. Test error: 3.31 pixels

My results were (Run with DLC *2.1.9* on Jan 2021) for 10 k iterations

**ResNet50
Imgaug:

Scalecrop:

Tensorpack:


**EffNet-b3
Imgaug:

Scalecrop:

Tensorpack:

Notice: despite the higher RMSE for imgaug due to the augmentation,
the network performs much better on the testvideo (see Neuron Primer: https://www.cell.com/neuron/pdf/S0896-6273(20)30717-0.pdf)

"""


import os
from pathlib import Path

os.environ["DLClight"] = "True"

import deeplabcut
import numpy as np

# Loading example data set
path_config_file = os.path.join(os.getcwd(), "openfield-Pranav-2018-10-30/config.yaml")
cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
maxiters = 10000

deeplabcut.load_demo_data(path_config_file)

## Create one split and make Shuffle 2 and 3 have the same split.
###Note that the new function in DLC 2.1 simplifies network/augmentation comparisons greatly:
deeplabcut.create_training_model_comparison(
    path_config_file,
    num_shuffles=1,
    net_types=["resnet_50", "efficientnet-b3"],
    augmenter_types=["imgaug", "scalecrop", "tensorpack"],
)


## here is an "old way" to do this
"""
trainIndices, testIndices=deeplabcut.mergeandsplit(path_config_file,trainindex=0,uniform=True)
deeplabcut.create_training_dataset(path_config_file,Shuffles=[2],trainIndices=trainIndices,testIndices=testIndices)
deeplabcut.create_training_dataset(path_config_file,Shuffles=[3],trainIndices=trainIndices,testIndices=testIndices)
for shuffle in [2,3]:
	if shuffle==3:
		posefile=os.path.join(cfg['project_path'],'dlc-models/iteration-'+str(cfg['iteration'])+'/'+ cfg['Task'] + cfg['date'] + '-trainset' + str(int(cfg['TrainingFraction'][0] * 100)) + 'shuffle' + str(shuffle),'train/pose_cfg.yaml')

		DLC_config=deeplabcut.auxiliaryfunctions.read_plainconfig(posefile)
		DLC_config['dataset_type']='tensorpack'
		deeplabcut.auxiliaryfunctions.write_plainconfig(posefile,DLC_config)
"""

for shuffle in 1 + np.arange(6):

    posefile, _, _ = deeplabcut.return_train_network_path(
        path_config_file, shuffle=shuffle
    )

    if shuffle % 3 == 1:  # imgaug
        edits = {"rotation": 180, "motion_blur": True}
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

    elif shuffle % 3 == 0:  # Tensorpack:
        edits = {"rotation": 180, "noise_sigma": 0.01}
        DLC_config = deeplabcut.auxiliaryfunctions.edit_config(posefile, edits)

    print("TRAIN NETWORK", shuffle)
    deeplabcut.train_network(
        path_config_file,
        shuffle=shuffle,
        saveiters=10000,
        displayiters=200,
        maxiters=maxiters,
        max_snapshots_to_keep=11,
    )

    print("EVALUATE")
    deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle], plotting=True)

    print("Analyze Video")

    videofile_path = os.path.join(
        os.getcwd(), "openfield-Pranav-2018-10-30", "videos", "m3v1mp4.mp4"
    )

    deeplabcut.analyze_videos(path_config_file, [videofile_path], shuffle=shuffle)

    print("Create Labeled Video and plot")
    deeplabcut.create_labeled_video(path_config_file, [videofile_path], shuffle=shuffle)
    deeplabcut.plot_trajectories(path_config_file, [videofile_path], shuffle=shuffle)
